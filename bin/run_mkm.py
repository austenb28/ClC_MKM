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
	mkm_sys = MKM_sys(configs[0])
	print('Ion flows:')
	print(mkm_sys.ion_flows)

main()
