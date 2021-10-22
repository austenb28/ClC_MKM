from ClC_MKM.src.MKM_sys import MKM_sys
import ClC_MKM.src.Config as conf
import numpy as np
import math

# Used to handle objective calculation
class Objective_handler():
	def __init__(self,mkm_configs,opt_config):
		self.n_sys = len(mkm_configs)
		self.residual_params = self.gen_residual_params(
			opt_config['opt_residuals_file'])
		self.n_residual_params = len(self.residual_params)
		self.mkm_systems = []
		self.mkm_systems.append(MKM_sys(mkm_configs[0]))
		self.coeffs = self.mkm_systems[0].coeffs
		self.ion_flows = [self.mkm_systems[0].ion_flows]
		for j in range(1,self.n_sys):
			self.mkm_systems.append(MKM_sys(mkm_configs[j]))
			self.mkm_systems[-1].coeffs = self.coeffs
			self.ion_flows.append(self.mkm_systems[-1].ion_flows)
		self.flow_alias_dict = self.gen_flow_alias_dict()
		# self.sq_residuals = self.gen_sq_residuals()

	# Returns the square residual for residual_parameter k, system j
	def get_sq_residual(self,j,k):
		return (self.flow_alias_dict[self.residual_params[k]](j) -
				getattr(self,self.residual_params[k])[j])**2

	# Returns flow_alias_dict which contains lambda functions for
	# corresponding flow calculations
	def gen_flow_alias_dict(self):
		flow_alias_dict = dict()
		flow_alias_dict['net_Cl_flow'] = lambda j:(
			self.ion_flows[j][0][0] + self.ion_flows[j][1][0])
		flow_alias_dict['net_H_flow'] = lambda j:(
			self.ion_flows[j][0][1] + self.ion_flows[j][1][1])
		flow_alias_dict['opp_Cl_flow'] = lambda j:(
			self.ion_flows[j][1][0])
		flow_alias_dict['opp_H_flow'] = lambda j:(
			self.ion_flows[j][1][1])
		flow_alias_dict['bio_Cl_flow'] = lambda j:(
			self.ion_flows[j][0][0])
		flow_alias_dict['bio_H_flow'] = lambda j:(
			self.ion_flows[j][0][1])
		return flow_alias_dict

	# Returns the residual_params list of strings.
	# Also initializes the corresponding attribute's
	# systems vector
	def gen_residual_params(self,filename):
		residual_params = []
		self.res_configs = conf.read_config_RES(
			filename,self.n_sys)
		for key in self.res_configs[0]:
			vec = np.zeros((self.n_sys))
			for j in range(self.n_sys):
				vec[j] = self.res_configs[j][key]
			setattr(self,key,vec)
			residual_params.append(key)
		return residual_params

	# Returns the objective using the specified flow
	# residuals and self.coeffs
	def evaluate(self):
		objective = 0
		for j in range(self.n_sys):
			self.mkm_systems[j].update_all()
			for k in range(self.n_residual_params):
				if math.isnan(self.flow_alias_dict[
					self.residual_params[k]](j)):
					return float('NaN')
				sq_residual = self.get_sq_residual(j,k)
				if not math.isnan(sq_residual):
					objective += sq_residual
		return objective