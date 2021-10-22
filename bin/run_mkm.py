#!/usr/bin/env python3

import argparse
import numpy as np
import math
import copy
from matplotlib import pyplot as plt
from ClC_MKM.src.MKM_sys import MKM_sys
import ClC_MKM.src.Config as conf

# Writes a numpy matrix mat to file filename.
def print_mat(mat,filename):
	bufflist = []
	for j in range(mat.shape[0]):
		for k in range(mat.shape[1]):
			if mat[j,k] == 0:
				exponent = 0
			else:
				exponent = int(math.log(abs(mat[j,k]),10))
			bufflist.append('{:.15f}E{:d}\t'.format(
				mat[j,k] / (10**exponent),
				exponent
			))
		bufflist.append('\n')
	with open(filename,'w') as myfile:
		myfile.write(''.join(bufflist))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('config',
		help='Configuration file (.txt)',
	)
	args = parser.parse_args()
	configs = conf.read_config_MKM(args.config)
	mkm_sys = MKM_sys(configs[0])
	print('Ion flows:')
	print(mkm_sys.ion_flows)
	# N = 1500
	# data_freq = 1
	# print(N//data_freq)
	# data = np.zeros([N//data_freq,8])
	# k = 0
	# for j in range(N):
	# 	mkm_opt.step()
	# 	if j % data_freq == 0:
	# 		data[k,:] = (
	# 			[j,mkm_opt.alpha,mkm_opt.objective,mkm_sys.ratio] + 
	# 			list(mkm_sys.ion_flows[0]) + 
	# 			list(mkm_sys.ion_flows[1]))
	# 		print('{:<4d} obj: {:<6.1E} alpha: {:<6.1E} improvement: {:<6.1E}'.format(
	# 			j, mkm_opt.objective, mkm_opt.alpha, mkm_opt.improvement
	# 		))
	# 		k += 1
	# print_mat(data,'output.dat')
	# mkm_sys.export_coeffs('out_coeffs.csv')
main()
