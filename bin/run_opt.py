#!/usr/bin/env python3

import argparse
import numpy as np
import math
import copy
from matplotlib import pyplot as plt
from ClC_MKM.src.Optimizer import Optimizer

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('config',
		help='Configuration file (.txt)',
	)
	args = parser.parse_args()
	mkm_opt = Optimizer(args.config)
	mkm_sys = mkm_opt.objective_handler.mkm_systems[0]
	print('Initial ion flows:')
	print(mkm_sys.ion_flows)
	mkm_opt.run()
main()
