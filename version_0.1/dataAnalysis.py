
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from pandas.plotting import scatter_matrix



import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import dataLoad as dl

def analysis():
	passenger = dl.loadData()

	print(passenger.head())
	#[5 rows x 12 columns]

	print(passenger.info())
	#Age            714 non-null float64
	#Cabin          204 non-null object
	#Embarked       889 non-null object

	print(passenger.describe())

	corr_matrix = passenger.corr()
	print(corr_matrix['Survived'])


	passenger.hist(bins=200,figsize=(20,15))
	dl.save_fig("hist_plot")
	#plt.show()

	attributes = ["Survived", "Pclass", "Fare"]
	scatter_matrix(passenger[attributes], figsize=(12, 8))
	dl.save_fig("scatter_matrix_plot")
	#plt.show()

	passenger.plot(kind="scatter", x="Pclass", y="Survived",
             alpha=0.1)
	plt.axis([0, 3, 0, 1])
	dl.save_fig("survived_vs_pclass_scatterplot")

	plt.show()








