#preTreatment.py

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

'''
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
'''

import dataLoad as dl


def preProcessData():
	passenger = dl.loadData()
	test_set = dl.loadData('test.csv')

	train_df = passenger.copy()
	test_df = test_set.copy()
	combine = [train_df, test_df]

	train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
	test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
	combine = [train_df, test_df]

	for dataset in combine:
		dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

	for dataset in combine:
		dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
		dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
		dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
		dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


	title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
	for dataset in combine:
		dataset['Title'] = dataset['Title'].map(title_mapping)
		dataset['Title'] = dataset['Title'].fillna(0)

	train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
	test_df = test_df.drop(['Name'], axis=1)
	combine = [train_df, test_df]

	for dataset in combine:
		dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)


	guess_ages = np.zeros((2,3))

	for dataset in combine:
		for i in range(0, 2):
			for j in range(0, 3):
				guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
				age_guess = guess_df.median()
				guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
		for i in range(0, 2):
			for j in range(0, 3):
				dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1), 'Age'] = guess_ages[i,j]
		dataset['Age'] = dataset['Age'].astype(int)

	train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
	train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

	for dataset in combine:
		dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
		dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
		dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
		dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
		dataset.loc[ dataset['Age'] > 64, 'Age']

	train_df = train_df.drop(['AgeBand'], axis=1)
	combine = [train_df, test_df]

	for dataset in combine:
		dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

	for dataset in combine:
		dataset['IsAlone'] = 0
		dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

	train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
	test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
	combine = [train_df, test_df]

	for dataset in combine:
		dataset['Age*Class'] = dataset.Age * dataset.Pclass

	freq_port = train_df.Embarked.dropna().mode()[0]

	for dataset in combine:
		dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
	for dataset in combine:
		dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

	test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

	train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
	for dataset in combine:
		dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
		dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
		dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
		dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
		dataset['Fare'] = dataset['Fare'].astype(int)

	train_df = train_df.drop(['FareBand'], axis=1)
	combine = [train_df, test_df]

	X_train = train_df.drop("Survived", axis=1)
	Y_train = train_df["Survived"]
	X_test  = test_df.drop("PassengerId", axis=1).copy()

	return X_train,Y_train,test_set,X_test

	