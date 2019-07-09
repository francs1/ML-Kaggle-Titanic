#modules.py

import numpy as np
import pandas as pd
import matplotlib as mpl
import sklearn


def checkVersion():
	print('numpy version : ' + np.__version__)#1.16.2
	print('pandas version : ' + pd.__version__)#0.24.2
	print('matplotlib version : ' + mpl.__version__)#3.0.3
	print('sk-learn version : ' + sklearn.__version__)#0.20.3