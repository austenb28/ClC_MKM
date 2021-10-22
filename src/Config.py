import re

# Produces dictionary config_args with format
# config_args[identifier] = (Required, type).
# config_args contains recognized parameters for
# MKM config files
def gen_config_args_MKM():
	config_args = dict()
	config_args['input_rate_file'] = (True, str)
	config_args['rate_map_file'] = (True, str)
	config_args['internal_pH'] = (True, float)
	config_args['external_pH'] = (True, float)
	config_args['internal_Cl_conc'] = (True, float)
	config_args['external_Cl_conc'] = (True, float)
	config_args['enzyme_MW'] = (True, float)
	config_args['lipid_MW'] = (True, float)
	config_args['area_per_lipid'] = (True, float)
	config_args['enzyme_lipid_wtfrac'] = (True, float)
	config_args['h_rxn_bl'] = (True, float)
	config_args['diffusivity_Cl'] = (True, float)
	config_args['diffusivity_H'] = (True, float)
	config_args['vesicle_diam'] = (True, float)
	config_args['enzyme_surf_conc_sim'] = (True, float)
	config_args['opt_config_file'] = (False, str)
	n_required = 0
	for arg,properties in config_args.items():
		if properties[0]:
			n_required += 1
	return config_args,n_required

# Produces dictionary config_args with format
# config_args[identifier] = (Required, type).
# config_args contains recognized parameters for
# optimization config files
def gen_config_args_OPT():
	config_args = dict()
	config_args['opt_residuals_file'] = (True, str)
	config_args['output_interval'] = (False, int)
	config_args['n_steps'] = (False, int)
	config_args['limp'] = (False, float)
	config_args['uimp'] = (False, float)
	config_args['max_limp_steps'] = (False, int)
	config_args['opt_dat_file'] = (False,str)
	#SciPy
	config_args['global_method'] = (False,str)
	config_args['local_method'] = (False,str)
	config_args['opt_package'] = (False,str)
	config_args['local_options_file'] = (False,str)
	config_args['global_options_file'] = (False,str)
	#SpotPy
	config_args['algorithm'] = (False,str)
	n_required = 0
	for arg,properties in config_args.items():
		if properties[0]:
			n_required += 1
	return config_args,n_required

# Produces dictionary config_args with format
# config_args[identifier] = (Required, type).
# config_args contains recognized parameters for
# optimization config files
def gen_config_args_RES():
	config_args = dict()
	config_args['net_Cl_flow'] = (False, float)
	config_args['net_H_flow'] = (False,float)
	config_args['bio_Cl_flow'] = (False, float)
	config_args['bio_H_flow'] = (False,float)
	config_args['opp_Cl_flow'] = (False, float)
	config_args['opp_H_flow'] = (False, float)
	n_required = 0
	for arg,properties in config_args.items():
		if properties[0]:
			n_required += 1
	return config_args,n_required

# Returns the number of system configurations specified in 
# The config file string
def get_n_sys(buff):
	n_fields = 1
	for line in buff:
		if line[0] == '#':
			continue
		line = re.split('[\t =\n]+', ' ' + line)
		line = line[1:]
		fields = re.split(',',line[1])
		l_fields = len(fields)
		if n_fields != 1 and l_fields != 1 and l_fields != n_fields:
			raise AssertionError("Inconsistent number of system configurations for config argument \"{:s}\".".format(
				line[0]
			))
		if l_fields > n_fields:
			n_fields = l_fields
	return n_fields

# Produces a list of dictionary configs with format 
# config[identifier] = value
# using MKM config file filename with format consistent
# with function gen_config_args_MKM()
def read_config_MKM(filename):
	return read_config(filename,gen_config_args_MKM)

# Produces a dictionary config with format 
# config[identifier] = value
# residual optimization config file filename with format 
# consistent with function gen_config_args_OPT()
def read_config_OPT(filename):
	return read_config(filename,gen_config_args_OPT,1)[0]

# Produces a list of dictionary configs with format 
# config[identifier] = value
# residual config file filename with format 
# consistent with function gen_config_args_RES()
def read_config_RES(filename,n_sys):
	return read_config(filename,gen_config_args_RES,n_sys)

def get_generic_properties(field):
	if field == 'True' or field == 'False':
		return(False,bool)
	int_fields = {'disp','maxiter','maxfev','maxcor',
	'maxfun','iprint','maxls','maxCGit','niter',
	'niter_success','seed','popsize','workers','iters',
	'maxev','minhgrd','local_iter'}
	if field in int_fields:
		return(False,int)
	try:
		float(field)
	except:
		return(False,str)
	return(False,float)

# Produces a list of dictionary configs with format 
# config[identifier] = value
# using config file filename with format consistent
# with a corresponding function gen_config_args()
def read_config(filename,gen_config_args=None,n_sys=None):
	if gen_config_args is None:
		n_required = 0
	else:
		config_args,n_required = gen_config_args()
	with open(filename,'r') as myfile:
		buff = myfile.readlines()
	if n_sys is None:
		n_sys = get_n_sys(buff)
	configs = [dict() for j in range(n_sys)]
	for line in buff:
		if line[0] == '#':
			continue
		line = re.split('[\t =\n]+', ' ' + line)
		line = line[1:]
		if (gen_config_args is not None and 
			line[0] not in config_args):
			raise AssertionError("Unrecognized config argument \"{:s}\" specified.".format(
				line[0]
			))
		fields = re.split(',',line[1])
		if gen_config_args is None:
			properties = get_generic_properties(fields[0])
		else:
			properties = config_args[line[0]]
		if len(fields) < n_sys:
			fields = fields*n_sys
		for j in range(n_sys):
			configs[j][line[0]] = properties[1](fields[j])
		if properties[0]:
			n_required -= 1
	if n_required > 0:
		raise AssertionError("{:d} unspecified required config arguments.".format(
			n_required
		))
	return configs