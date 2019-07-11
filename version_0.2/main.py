from __future__ import division, print_function, unicode_literals

#system module
import sys
import os
import random

#common mechine learning tools and frameworks
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sb
import sklearn as skl

#custom module
from constantInit import *
import modules

import dataAnalysis as da
import trainingModel as tm
import preTreatment as pt


def main():
	#modules.checkVersion()
	#da.analysis()

	#pt.preProcessData()

	#tm.trainSGD()
	#tm.trainLR()
	#tm.trainDT()
	tm.trainRF()
	
if __name__ == '__main__':
	main()