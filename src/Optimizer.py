import re
import numpy as np
import math
import copy
from ClC_MKM.src.Objective_handler import Objective_handler
import ClC_MKM.src.Config as conf
import ClC_MKM.src.Output as outp
from ClC_MKM.src.Grad_CG import Grad_CG
from ClC_MKM.src.SciPy_opt import SciPy_opt
from ClC_MKM.src.SpotPy_opt import SpotPy_opt

# Used to perform optimization for the ClC-ec1 MKM system
class Optimizer():
	def __init__(self,filename):
		self.mkm_configs = conf.read_config_MKM(filename)
		self.opt_config = conf.read_config_OPT(
			self.mkm_configs[0]['opt_config_file'])
		self.objective_handler = Objective_handler(
			self.mkm_configs,self.opt_config)
		self.objective = 0
		self.update_objective()
		self.prev_objective = self.objective
		self.coeffs = self.objective_handler.coeffs
		self.prev_coeffs = np.copy(self.coeffs)
		self.gradient = np.copy(self.coeffs)
		self.N_coeffs = len(self.coeffs)
		self.coeff_deltas = np.copy(self.coeffs)
		self.lbounds = np.zeros(self.N_coeffs)
		# self.ubounds = np.full(self.N_coeffs,float('inf'))
		self.ubounds = self.coeffs*10
		self.min_delta = 1E-6
		self.sub_opt = self.init_sub_opt()
		self.output_interval = self.opt_config['output_interval']
		self.n_steps = self.opt_config['n_steps']
		self.opt_dat = self.init_opt_dat()
		self.out_count = 1
		if 'opt_dat_file' in self.opt_config:
			self.opt_dat_file = self.opt_config['opt_dat_file']
		else:
			self.opt_dat_file = 'opt.dat'

	# Returns an initialized the output data matrix storing
	# coefficients and the objective
	def init_opt_dat(self):
		opt_dat = np.zeros((1+self.n_steps//self.output_interval,
			self.N_coeffs+2))
		opt_dat[0,1:-1] = self.coeffs
		opt_dat[0,-1] = self.objective
		return opt_dat

	# Returns the appropriate sub_opt class using self.opt_config
	def init_sub_opt(self):
		if 'opt_package' in self.opt_config:
			opt_package = self.opt_config['opt_package']
		else:
			opt_package = None
		if opt_package is None:
			sub_opt = Grad_CG(self)
		elif opt_package == 'scipy':
			sub_opt = SciPy_opt(self)
		elif opt_package == 'spotpy':
			sub_opt = SpotPy_opt(self)
		return sub_opt

	# Updates self.coeff_deltas for numerical differentiation
	def update_coeff_deltas(self):
		for j in range(self.N_coeffs):
			self.coeff_deltas[j] = self.calc_coeff_delta(j)

	# Returns the differentiation delta for coeff j
	def calc_coeff_delta(self,j):
		precision = 1E-4
		delta = abs(self.prev_coeffs[j])*precision
		if delta < self.min_delta:
			delta = self.min_delta
		return delta

	# Returns the partial derivative of the objective with respect
	# to coeff j
	# NOTE: this invalidates flow matrices, ion flows, and ratios
	def calc_partial(self,j):
		delta = self.coeff_deltas[j]
		self.coeffs[j] = self.prev_coeffs[j] + delta
		self.update_objective()
		delta_obj = self.objective - self.prev_objective
		self.coeffs[j] = self.prev_coeffs[j]
		deriv = delta_obj/delta
		#TODO: probably remove conditional below for SciPy jac calculations
		if ((self.prev_coeffs[j] == self.lbounds[j] and deriv > 0) or
			self.prev_coeffs[j] == self.ubounds[j] and deriv < 0):
			deriv = 0
			# print("Maintaining constraint on coefficient {:d}".format(j))
		# if self.step_num == self.debug_num and j == 3:
		# 	print(j,self.coeffs[j],deriv)
		# 	quit()
		return deriv

	# Updates self.gradient to contain the objective's gradient
	#TODO: potentially update such that explicit relative
	# objective deltas are achieved
	def update_gradient(self):
		self.prev_objective = self.objective
		self.update_coeff_deltas()
		for j in range(self.N_coeffs):
			self.gradient[j] = self.calc_partial(j)
		self.objective = self.prev_objective

	# Updates self.objective using self.coeffs
	def update_objective(self):
		self.objective = self.objective_handler.evaluate()

	# Saves output data
	def save_dat(self):
		header = ('Step\t' + '\t'.join( 
			self.objective_handler.mkm_systems[0].idents) +
			' Objective')
		np.savetxt(self.opt_dat_file,self.opt_dat,fmt='%.6e',
			header=header,comments='',delimiter='\t')
		self.objective_handler.mkm_systems[0].export_coeffs(
			'out_coeffs.csv')

	# Runs the optimization
	def run(self):
		self.sub_opt.run()
		self.save_dat()
