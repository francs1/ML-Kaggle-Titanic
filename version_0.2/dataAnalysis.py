
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import seaborn as sns

from pandas.plotting import scatter_matrix

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

import dataLoad as dl
import preTreatment as pt

def analysis():
	passenger = dl.loadData()

	print(passenger.head())
	#[5 rows x 12 columns]

	print(passenger.info())
	#Age            714 non-null float64
	#Cabin          204 non-null object
	#Embarked       889 non-null object

	print(passenger.describe())
	print(passenger.describe(include=['O']))

	print(passenger[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
	print(passenger[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
	print(passenger[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
	print(passenger[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))


	corr_matrix = passenger.corr()
	print(corr_matrix['Survived'])

	g = sns.FacetGrid(passenger, col='Survived')
	g.map(plt.hist, 'Age', bins=20)


	grid = sns.FacetGrid(passenger, col='Survived', row='Pclass', size=2.2, aspect=1.6)
	grid.map(plt.hist, 'Age', alpha=.5, bins=20)
	grid.add_legend();
	
	
	grid = sns.FacetGrid(passenger, row='Embarked', size=2.2, aspect=1.6)
	grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
	grid.add_legend()


	grid = sns.FacetGrid(passenger, row='Embarked', col='Survived', size=2.2, aspect=1.6)
	grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
	grid.add_legend()
	

	'''
	##show plot:
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

	
	'''
	plt.show()







