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
	strat_passenger_set = passenger.copy()
	passenger_data = strat_passenger_set.drop(['PassengerId','Survived','Name','Sex','Ticket','Cabin','Embarked'],axis=1)
	passenger_labels = strat_passenger_set[['Survived']].copy()


	test_set = dl.loadData('test.csv')
	test_data = test_set.drop(['PassengerId','Name','Sex','Ticket','Cabin','Embarked'],axis=1)


	# #process non value-->

	# #age_median = passenger['Age'].median()
	# #passenger['Age'] = passenger['Age'].fillna(age_median)
	# #print(passenger.info())

	# imputer = SimpleImputer(strategy="median")
	# passenger_data = pd.DataFrame(imputer.fit_transform(passenger_data.values),columns=passenger_data.columns)


	# #object value to one-hot encode-->

	# passenger_sex = passenger[['Sex']]
	# cat_encoder = OneHotEncoder()
	# passenger_cat_1hot = cat_encoder.fit_transform(passenger_sex).toarray()
	# print(type(passenger_cat_1hot))

	sex_pipeline = Pipeline([
		('one_hot',OneHotEncoder()),
	])


	num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

	num_attribs = list(passenger_data.columns)
	cat_attribs = ['Sex']


	full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", sex_pipeline, cat_attribs),
    ])

	passenger_prepared = full_pipeline.fit_transform(strat_passenger_set)

	passenger_test = full_pipeline.fit_transform(test_set)

	#print(passenger_prepared)

	# #train_set, test_set = dl.split_train_test_by_id(passenger, 0.2,'PassengerId')
	# #print(len(train_set), "train +", len(test_set), "test")

	# #train_set, test_set = train_test_split(passenger_prepared, test_size=0.2, random_state=42)
	# #print(len(train_set), "train +", len(test_set), "test")

	return passenger_prepared,passenger_labels,test_set,passenger_test