import re
import numpy as np
import math
import copy
from matplotlib import pyplot as plt

# Used to represent the ClC-ec1 MKM system
class MKM_sys():
	def __init__(self,config):
		self.surf_conc_mode = 'uniform'
		self.flow_transitions = self.gen_flow_transitions()
		self.h_rxn_bl = config['h_rxn_bl']
		self.concs = self.get_concs(config)
		self.coeffs_dict = self.get_coeffs_dict(config)
		self.init_coeffs = np.array(
			self.dict_to_list(self.coeffs_dict))
		self.idents = list(self.coeffs_dict)
		self.transitions = self.gen_transitions(config)
		self.n_uptakes = np.zeros((2,2),dtype=int)
		self.n_releases = np.zeros((2,2),dtype=int)
		self.uptake_transitions = self.gen_surf_transitions(True)
		self.release_transitions = self.gen_surf_transitions(False)
		self.AVOGADRO = 6.02214E23
		self.PI = 3.14159
		vesicle_area = self.PI * config['vesicle_diam']**2
		self.enzymes_per_vesicle = (
			config['enzyme_lipid_wtfrac'] /
			config['enzyme_MW'] *
			config['lipid_MW'] /
			config['area_per_lipid'] *
			vesicle_area)
		if self.enzymes_per_vesicle < 2:
			self.enzymes_per_vesicle = 2
			self.enzyme_surf_conc = (
				self.enzymes_per_vesicle /
				vesicle_area /
				self.AVOGADRO
			)
		else:
			self.enzyme_surf_conc = (
				config['enzyme_lipid_wtfrac'] /
				config['enzyme_MW'] * 
				config['lipid_MW'] / 
				self.AVOGADRO /
				config['area_per_lipid'])
		# print(self.enzymes_per_vesicle,
		# 	self.enzyme_surf_conc)
		self.enzyme_surf_conc_sim = config['enzyme_surf_conc_sim']
		if self.surf_conc_mode != 'uniform':
			self.initialize_surf_sums()
		self.coeffs = np.copy(self.init_coeffs)
		self.N_coeffs = len(self.coeffs)
		self.N_sites = 6
		self.N_states = 2**self.N_sites
		self.N_orient = 2
		self.coeff_mats = np.zeros((self.N_orient,self.N_states,
			self.N_states))
		self.As = np.copy(self.coeff_mats)
		self.As[:,-1,:] = 1
		self.b = np.zeros([self.N_states,1])
		self.b[-1] = 1
		self.ss_pops = np.zeros((self.N_states,self.N_orient))
		self.flow_mats = np.copy(self.coeff_mats)
		self.ion_flows = np.zeros((2,2))
		self.ratio = 0
		self.expanded_coeff = np.zeros(2)
		if self.surf_conc_mode == 'uniform':
			self.surf_concs = np.copy(self.concs)
		elif self.enzymes_per_vesicle == 2:
			self.surf_concs = np.zeros((2,2,2))
		else:
			self.surf_concs = np.zeros((2,2))
		self.mxfer_coeffs = self.gen_mxfer_coeffs(config)
		# self.update_all()
		self.update_all()
		# print('Ion flows:\n',self.ion_flows)
		# print(self.ss_pops)
		# print(np.sum(self.ss_pops[:,0]),
		# 	np.sum(self.ss_pops[:,1]))

	# Writes self.coeffs to filename
	def export_coeffs(self,filename):
		bufflist = ['"cl0_-00","cl1_-00","cl0_00+","cl1_00+","cl0_+00","cl1_+00","cl0_00-","cl1_00-","cl0_+01","cl1_+01","cl0_10+","cl1_10+","cl0_-01","cl1_-01","cl0_10-","cl1_10-","cl0_01+","cl1_01+","cl0_+10","cl1_+10","cl0_-10","cl1_-10","cl0_01-","cl1_01-","cl0_+11","cl1_+11","cl0_-11","cl1_-11","cl0_11+","cl1_11+","cl0_11-","cl1_11-","cl0_0+-","cl1_0+-","cl0_0-+","cl1_0-+","cl1_+-0","cl1_+-1","cl1_-+0","cl1_-+1","cl0_1+-","cl1_1+-","cl0_1-+","cl1_1-+","203p0","203p1","203d0","203d1","148d00","148d01","148p00","148p01","148d10","148d11","148p10","148p11","h+-00_","h-+00_","h+-01_","h-+01_","h+-10_","h-+10_","h+-11_","h-+11_","u1__0_","d1__0_","u0__0_","d0__0_","u1__1_"\n']
		for coeff in self.coeffs:
			if coeff == 0:
				exponent = 0
			else:
				exponent = int(math.log(abs(coeff),10))
			bufflist.append('{:.15f}E{:d}\t'.format(
				coeff / (10**exponent),exponent
			))
			bufflist.append(',')
		bufflist[-1] = '\n'
		with open(filename,'w') as myfile:
			myfile.write(''.join(bufflist))

	# Updates all MKM system properties of self
	def update_all(self):
		self.update_ss_pops()
		self.update_flow_mats()
		self.update_ion_flows()
		self.update_ratio()

	# Updates self.flow_mats using self.ss_pops
	def update_flow_mats(self):
		np.copyto(self.flow_mats,self.coeff_mats)
		for j in range(self.N_orient):
			np.fill_diagonal(self.flow_mats[j], 0)
			for k in range(self.N_states):
				self.flow_mats[j,:,k] *= self.ss_pops[k,j]

	# Updates self.ss_pops using self.coeffs
	# TODO: allow repeated iteration calculations
	def update_ss_pops(self):
		niter = 1
		for j in range(niter):
			self.update_surf_concs()
			self.update_coeff_mats()
			self.iterate_ss_pops()
			# print(self.ss_pops[:3,:])
			# print_mat(self.coeff_mats[0],'test.dat')
		# self.update_surf_concs()
		# print('Surface concentrations:\n',self.surf_concs)

	# Updates self.ss_pops using self.coeffs via 1 iteration
	def iterate_ss_pops(self):
		for j in range(self.N_orient):
			# print(self.ss_pops[0])
			try:
				self.ss_pops[:,j] = np.ravel(
					np.linalg.solve(self.As[j],self.b))
			except np.linalg.LinAlgError as e:
				if str(e) == 'Singular matrix':
					print('Warning: Singular matrix encountered. Invalidating populations.')
					self.ss_pops[:,j] = float('NaN')

	# Returns mass transfer coefficient matrix mxfer_coeffs from
	# Input data
	def gen_mxfer_coeffs(self,config):
		mxfer_coeffs = np.zeros((2,2))
		diffusivities = (config['diffusivity_Cl'],config['diffusivity_H'])
		Shs = (3.29,2)
		for j in range(2):
			for k in range(2):
				mxfer_coeffs[j,k] = (Shs[j] * 
					diffusivities[k]/config['vesicle_diam'])
		return mxfer_coeffs

	# Returns tuple containing surface transition indices
	# with format (location=0(int) or 1 (ext),ion=0(Cl) or 1(H))
	def get_surf_indices(self, transition, uptake):
		if uptake:
			j1 = '0'
			j2 = '1'
		else:
			j1 = '1'
			j2 = '0'
		keys = [
			(1,2,1), # (binary_index, adjacent_index, ion(0=Cl,1=H))
			(2,1,1),
			(3,4,0),
			(5,4,0)
		]
		s1 = self.decode(transition[0])
		s2 = self.decode(transition[1])
		for key in keys:
			if (s1[key[0]] == j1 and s2[key[0]] == j2 and
				s1[key[1]] == s2[key[1]]):
				if key[1] > key[0]:
					return (1,key[2])
				else:
					return (0,key[2])
		return tuple()

	# Generates surface transitions dict
	# with format [[Cl_int,H_int],[Cl_ext,H_ext]]
	def gen_surf_transitions(self,uptake):
		surf_transitions = [[dict(),dict()],[dict(),dict()]]
		for j,transitions in enumerate(self.transitions):
			indices = self.get_surf_indices(transitions[0],uptake)
			if len(indices) != 0:
				surf_transitions[indices[0]][indices[1]][j] = transitions
				if uptake:
					self.n_uptakes[indices[0]][indices[1]] += len(transitions)
				else:
					self.n_releases[indices[0]][indices[1]] += len(transitions)
		return surf_transitions

	# Initializes surface concentratoin summation vectors
	# with format [orient][int/ext][ion] or [int/ext][ion]
	def initialize_surf_sums(self):
		initializer = [[[],[]],[[],[]]]
		self.uptake_coeffs = copy.deepcopy(initializer)
		self.release_coeffs = copy.deepcopy(initializer)
		initializer = [
			[[[],[]],[[],[]]],
			[[[],[]],[[],[]]]
		]
		self.uptake_concs = copy.deepcopy(initializer)
		self.release_concs = copy.deepcopy(initializer)
		for j in range(2):
			for k in range(2):
				if j == 0:
					j_opp = 1
				else:
					j_opp = 0
				if self.enzymes_per_vesicle == 2:
					self.uptake_coeffs[j][k] = np.zeros(
						self.n_uptakes[j,k])
					self.release_coeffs[j][k] = np.zeros(
						self.n_releases[j,k])
				else:
					self.uptake_coeffs[j][k] = np.zeros(
						self.n_uptakes[j,k] +
						self.n_uptakes[j_opp,k])
					self.release_coeffs[j][k] = np.zeros(
						self.n_releases[j,k] +
						self.n_releases[j_opp,k])
				if self.enzymes_per_vesicle == 2:
					self.uptake_concs[0][j][k] = (
						np.zeros(self.n_uptakes[j,k]))
					self.uptake_concs[1][j][k] = (
						np.zeros(self.n_uptakes[j_opp,k]))
					self.release_concs[0][j][k] = (
						np.zeros(self.n_releases[j,k]))
					self.release_concs[1][j][k] = (
						np.zeros(self.n_releases[j_opp,k]))
				else:
					self.uptake_concs[j][k] = (
						np.copy(self.uptake_coeffs[j][k]))
					self.release_concs[j][k] = (
						np.copy(self.release_coeffs[j][k]))

	# Updates surface concentrations
	def update_surf_concs(self):
		if self.surf_conc_mode == 'uniform':
			return
		if self.enzymes_per_vesicle == 2:
			self.update_surf_concs_seperate()
		else:
			self.update_surf_concs_combo()

	# Updates surface concentrations for the case that
	# there are 2 enzyme monomers (1 dimer) per vesicle,
	# So bio/opposite oriented systems are seperated
	def update_surf_concs_seperate(self):
		for j in range(2):
			for k in range(2):
				if j == 0:
					j_opp = 1
				else:
					j_opp = 0
				i_up = 0
				for coeff_index in self.uptake_transitions[j][k]:
					for transition in self.uptake_transitions[j][k][coeff_index]:
						self.uptake_coeffs[j][k][i_up] = self.coeffs[coeff_index]
						self.uptake_concs[0][j][k][i_up] = self.ss_pops[transition[0],0]
						i_up += 1
				i_up = 0
				for coeff_index in self.uptake_transitions[j_opp][k]:
					for transition in self.uptake_transitions[j_opp][k][coeff_index]:
						self.uptake_concs[1][j][k][i_up] = self.ss_pops[transition[0],1]
						i_up += 1
				i_rel = 0
				for coeff_index in self.release_transitions[j][k]:
					for transition in self.release_transitions[j][k][coeff_index]:
						self.release_coeffs[j][k][i_rel] = self.coeffs[coeff_index]
						self.release_concs[0][j][k][i_rel] = self.ss_pops[transition[0],0]
						i_rel += 1
				i_rel = 0
				for coeff_index in self.release_transitions[j_opp][k]:
					for transition in self.release_transitions[j_opp][k][coeff_index]:
						self.release_concs[1][j][k][i_rel] = self.ss_pops[transition[0],1]
						i_rel += 1
				self.uptake_coeffs[j][k] /= self.enzyme_surf_conc_sim # TODO move this to initialization, as well as in the expanded_coeff function
				for l in range(self.N_orient):
					self.uptake_concs[l][j][k] *= self.enzyme_surf_conc
					self.release_concs[l][j][k] *= self.enzyme_surf_conc
					if l == 0:
						j_ind = j
					else:
						j_ind = j_opp
					s_rel = np.dot(
						self.release_coeffs[j_ind][k],
						self.release_concs[l][j][k])
					s_up = np.dot(self.uptake_coeffs[j_ind][k],
						self.uptake_concs[l][j][k])
					self.surf_concs[l,j,k] = ((self.mxfer_coeffs[j,k] * 
						self.concs[j,k] + s_rel) /(
						self.mxfer_coeffs[j,k] + self.h_rxn_bl * 
						s_up))

	# Updates surface concentrations for the case that
	# bio/opposite systems occur on the same vesicles
	def update_surf_concs_combo(self):
		for j in range(2):
			for k in range(2):
				if j == 0:
					j_opp = 1
				else:
					j_opp = 0
				i_up = 0
				i_rel = 0
				for coeff_index in self.uptake_transitions[j][k]:
					for transition in self.uptake_transitions[j][k][coeff_index]:
						self.uptake_coeffs[j][k][i_up] = self.coeffs[coeff_index]
						self.uptake_concs[j][k][i_up] = self.ss_pops[transition[0],0]
						i_up += 1
				for coeff_index in self.uptake_transitions[j_opp][k]:
					for transition in self.uptake_transitions[j_opp][k][coeff_index]:
						self.uptake_coeffs[j][k][i_up] = self.coeffs[coeff_index]
						self.uptake_concs[j][k][i_up] = self.ss_pops[transition[0],1]
						i_up += 1
				for coeff_index in self.release_transitions[j][k]:
					for transition in self.release_transitions[j][k][coeff_index]:
						self.release_coeffs[j][k][i_rel] = self.coeffs[coeff_index]
						self.release_concs[j][k][i_rel] = self.ss_pops[transition[0],0]
						i_rel += 1
				for coeff_index in self.release_transitions[j_opp][k]:
					for transition in self.release_transitions[j_opp][k][coeff_index]:
						self.release_coeffs[j][k][i_rel] = self.coeffs[coeff_index]
						self.release_concs[j][k][i_rel] = self.ss_pops[transition[0],1]
						i_rel += 1
				self.uptake_coeffs[j][k] /= self.enzyme_surf_conc_sim # TODO move this to initialization, as well as in the expanded_coeff function
				self.uptake_concs[j][k] *= self.enzyme_surf_conc
				self.release_concs[j][k] *= self.enzyme_surf_conc
				s_rel = np.dot(
					self.release_coeffs[j][k],
					self.release_concs[j][k])
				s_up = np.dot(self.uptake_coeffs[j][k],
					self.uptake_concs[j][k])
				self.surf_concs[j,k] = ((self.mxfer_coeffs[j,k] * 
					self.concs[j,k] + s_rel) /(
					self.mxfer_coeffs[j,k] + self.h_rxn_bl * 
					s_up))
				# frac_rel = s_rel/(self.mxfer_coeffs[j,k] * 
				# 	self.concs[j,k])
				# frac_up = self.h_rxn_bl*s_up/self.mxfer_coeffs[j,k]
				# print(j,k,frac_rel, frac_up)

	# Updates self.expanded_coeff using appropriate concentrations
	# if the coeff represents an uptake transition
	def update_expanded_coeff(self,j):
		self.expanded_coeff[:] = self.coeffs[j]
		if self.enzymes_per_vesicle == 2 and self.surf_conc_mode != 'uniform':
			for k in range(2):
				if j in self.uptake_transitions[0][k]:
					self.expanded_coeff[0] *= (self.h_rxn_bl*
						self.surf_concs[0,0,k]/self.enzyme_surf_conc_sim)
					self.expanded_coeff[1] *= (self.h_rxn_bl*
						self.surf_concs[1,1,k]/self.enzyme_surf_conc_sim)
					return
				elif j in self.uptake_transitions[1][k]:
					self.expanded_coeff[0] *= (self.h_rxn_bl*
						self.surf_concs[0,1,k]/self.enzyme_surf_conc_sim)
					self.expanded_coeff[1] *= (self.h_rxn_bl*
						self.surf_concs[1,0,k]/self.enzyme_surf_conc_sim)
					return
		else:
			for k in range(2):
				if j in self.uptake_transitions[0][k]:
					self.expanded_coeff[0] *= (self.h_rxn_bl*
						self.surf_concs[0,k]/self.enzyme_surf_conc_sim)
					self.expanded_coeff[1] *= (self.h_rxn_bl*
						self.surf_concs[1,k]/self.enzyme_surf_conc_sim)
					return
				elif j in self.uptake_transitions[1][k]:
					self.expanded_coeff[0] *= (self.h_rxn_bl*
						self.surf_concs[1,k]/self.enzyme_surf_conc_sim)
					self.expanded_coeff[1] *= (self.h_rxn_bl*
						self.surf_concs[0,k]/self.enzyme_surf_conc_sim)
					return

	# Produces rate matrices in a list with orientations
	# biological (self.coeff_mats[0]) and opposite (self.coeff_mats[1])
	# And associated solution matrices in self.As
	def update_coeff_mats(self):
		np.fill_diagonal(self.coeff_mats[0], 0)
		np.fill_diagonal(self.coeff_mats[1], 0)
		for j in range(self.N_coeffs):
			self.update_expanded_coeff(j)
			for t in self.transitions[j]:
				for k in range(self.N_orient):
					self.coeff_mats[k][t[1],t[0]]  = self.expanded_coeff[k]
					self.coeff_mats[k][t[0],t[0]] -= self.expanded_coeff[k]
		for k in range(self.N_orient):
			np.copyto(self.As[k,:-1,:],self.coeff_mats[k,:-1,:])

	# Produces orientational ion flows with format
	# [bio_flows,opp_flows] with subformat
	# [cl_flow,h_flow] with positive flows corresponding
	# to flow directed from extracellular to intracellular
	def update_ion_flows(self):
		for j in range(self.N_orient):
			for k in range(2):
				self.ion_flows[j,k] = 0
		for t in self.flow_transitions:
			self.ion_flows[0,t[2]] += (
				t[3]*self.flow_mats[0][t[1],t[0]]
			)
			self.ion_flows[1,t[2]] -= (
				t[3]*self.flow_mats[1][t[1],t[0]]
			)

	# Updates the ion flow ratio self.ratio using self.ion_flows
	def update_ratio(self):
		denominator = self.ion_flows[0,1] + self.ion_flows[1,1]
		if denominator == 0:
			self.ratio = float('NaN')
		else:
			self.ratio = (
				(self.ion_flows[0,0] + self.ion_flows[1,0])/
				denominator
		)

	# Produces the integer corresponding to the 
	# binary string ident.
	@staticmethod
	def encode(ident):
		N = len(ident)
		index = 0
		rev_indent = ident[::-1]
		for j in range(N):
			index += int(rev_indent[j]) * 2 ** j
		return index

	# Produces a binary string of length N corresponding to 
	# an integer index.
	@staticmethod
	def decode(index, N = 6):
		result = ['0' for _ in range(N)]
		for j in range(N)[::-1]:
			result[j] = str(index // 2 ** j)
			index -= int(result[j]) * 2 ** j
		result.reverse()
		return ''.join(result)

	# Produces flow transitions with format
	# [(state_1,state_2,ion,sign),...]
	# where ion format 0=Cl,1=H, and sign format 
	# 1 = positive flow (bio. orient), -1 = negative flow (bio. orient)
	def gen_flow_transitions(self):
		N = 64
		flow_transitions = []
		keys = [
			(1,1), # (binary_index, ion(0=Cl,1=H))
			(3,0)]
		for j in range(N):
			for k in range(N):
				for key in keys:
					s1 = self.decode(j)
					s2 = self.decode(k)
					s1_ext = s1[:key[0]] + s1[key[0]+1:]
					s2_ext = s2[:key[0]] + s2[key[0]+1:]
					if (s1[key[0]] == '0' and
							s2[key[0]] == '1' and
							s1_ext == s2_ext):
						flow_transitions.append(
							(j,k,key[1],1))
					elif (s1[key[0]] == '1' and
						s2[key[0]] == '0' and
						s1_ext == s2_ext):
						flow_transitions.append(
							(j,k,key[1],-1))
		return flow_transitions

	# Returns the transition corresponding to to the state strings
	# m1 -> m2
	def get_transitions(self,m1,m2):
		transitions = []
		N_states = len(m1)
		N_fill = 0
		for j in range(N_states):
			if m1[j] == 'X':
				N_fill += 1
		N = 2**N_fill
		for j in range(N):
			filler = self.decode(j,N_fill)
			s1 = []
			s2 = []
			k = 0
			for l in range(N_states):
				if m1[l] == 'X':
					s1.append(filler[k])
					s2.append(filler[k])
					k += 1
				else:
					s1.append(m1[l])
					s2.append(m2[l])
			transitions.append((
				self.encode(''.join(s1)),
				self.encode(''.join(s2))
			))
		return tuple(transitions)

	# Produces a rate_map dictionary with format
	# rate_map[(state_1, state_2, identifier)] = rate_coefficient.
	# 'X' in a state is a wildcard.
	# Units are 1/ms for first order and 1/ms/mM for second order
	def gen_transition_dict(self,config):
		with open(config['rate_map_file'],'r') as myfile:
			buff = myfile.readlines()
		lines = []
		for line in buff:
			line = re.split('[,\n]+', ',' + line + ',')
			lines.append(line[1:-1])
		transition_dict = dict()
		N = len(lines[1][0][2:])
		for line in lines[1:]:
			s1 = line[0][2:]
			s2 = line[1][2:]
			mask = line[2][2:]
			m1 = []
			m2 = []
			for j in range(N):
				if mask[j] == '0':
					m1.append('X')
					m2.append('X')
				else:
					m1.append(s1[j])
					m2.append(s2[j])
			transitions = self.get_transitions(m1,m2)
			ident = line[3]
			transition_dict[ident] = transitions
		return transition_dict

	#returns the list of transitions using config
	def gen_transitions(self,config):
		transition_dict = self.gen_transition_dict(config)
		return self.dict_to_list(transition_dict)

	# returns a list from a dict
	@staticmethod
	def dict_to_list(dict_in):
		list_out = []
		for ident in dict_in:
			list_out.append(dict_in[ident])
		return list_out

	# Produces dictionary rate_coeffs with format
	# rate_coeffs[identifier] = rate_coefficient.
	# Units are 1/ms for first order and 1/ms/mM for second order
	def get_coeffs_dict(self,config):
		with open(config['input_rate_file'],'r') as myfile:
			buff = myfile.readlines()
		lines = []
		for line in buff:
			line = re.split('[, \t"\n]+', ' ' + line + ' ')
			lines.append(line[1:-1])
		coeffs_dict = dict()
		for j, identfier in enumerate(lines[0]):
			coeffs_dict[identfier] = float(lines[1][j])
		return coeffs_dict

	# Produces internal and external concentrations with format
	# concentrations[[Cl_int,H_int],[Cl_ext,H_ext]]
	# Units are M (mol/m^3)
	def get_concs(self,config):
		concs = np.zeros((2,2))
		concs[0,0] = config['internal_Cl_conc']
		concs[1,0] = config['external_Cl_conc']
		concs[0,1] = 10**(-1*config['internal_pH']) * 1E3
		concs[1,1] = 10**(-1*config['external_pH']) * 1E3
		return concs

