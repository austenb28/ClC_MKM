#!/usr/bin/env python3

import argparse
import numpy as np
import math
import copy
from matplotlib import pyplot as plt
from ClC_MKM.src.MKM_sys import MKM_sys
import ClC_MKM.src.Config as conf

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('config',
		help='Configuration file (.txt)',
	)
	args = parser.parse_args()
	configs = conf.read_config_MKM(args.config)
	N = len(configs)
	for j in range(N):
		mkm_sys = MKM_sys(configs[j])
		print('System {:d} Ion flows (ions/ms):'.format(j))
		print('Bio Cl: {:>11.4e}'.format(mkm_sys.ion_flows[0][0]))
		print('Bio  H: {:>11.4e}'.format(mkm_sys.ion_flows[0][1]))
		print('Opp Cl: {:>11.4e}'.format(mkm_sys.ion_flows[1][0]))
		print('Opp  H: {:>11.4e}\n'.format(mkm_sys.ion_flows[1][1]))

main()
