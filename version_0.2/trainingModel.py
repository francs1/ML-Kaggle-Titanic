# trainingModel.py
import sys

import pandas as pd
import numpy as np

from sklearn.base import clone

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


from constantInit import *
import dataLoad as dl

import preTreatment as pt

np.random.seed(42)


def trainLR():
	passenger_prepared,passenger_labels,test_set,passenger_test = pt.preProcessData()

	lr_clf = LogisticRegression(C=1.0, penalty='l2', tol=1e-6, solver='lbfgs')
	passenger_labels = passenger_labels.values.reshape((len(passenger_labels.values),))
	lr_clf.fit(passenger_prepared, passenger_labels)

	print(cross_val_score(lr_clf, passenger_prepared, passenger_labels, cv=3, scoring="accuracy"))

	predict_data = lr_clf.predict(passenger_test)
	PassengerIds = test_set['PassengerId']
	results = pd.Series(predict_data,name="Survived",dtype=np.int32)
	submission = pd.concat([PassengerIds,results],axis = 1)
	dl.saveData(submission,'LRClassifier.csv')


def trainSGD():
	passenger_prepared,passenger_labels,test_set,passenger_test = pt.preProcessData()

	sgd_clf = SGDClassifier(max_iter=5, tol=-np.infty, random_state=42)
	passenger_labels = passenger_labels.values.reshape((len(passenger_labels.values),))
	sgd_clf.fit(passenger_prepared, passenger_labels)

	print(cross_val_score(sgd_clf, passenger_prepared, passenger_labels, cv=3, scoring="accuracy"))

	predict_data = sgd_clf.predict(passenger_test)
	PassengerIds = test_set['PassengerId']
	results = pd.Series(predict_data,name="Survived",dtype=np.int32)
	submission = pd.concat([PassengerIds,results],axis = 1)

	dl.saveData(submission,'SGDClassifier.csv')

def trainDT():
	passenger_prepared,passenger_labels,test_set,passenger_test = pt.preProcessData()

	sgd_clf = DecisionTreeClassifier()
	passenger_labels = passenger_labels.values.reshape((len(passenger_labels.values),))
	sgd_clf.fit(passenger_prepared, passenger_labels)

	print(cross_val_score(sgd_clf, passenger_prepared, passenger_labels, cv=3, scoring="accuracy"))

	predict_data = sgd_clf.predict(passenger_test)
	PassengerIds = test_set['PassengerId']
	results = pd.Series(predict_data,name="Survived",dtype=np.int32)
	submission = pd.concat([PassengerIds,results],axis = 1)

	dl.saveData(submission,'DTClassifier.csv')

def trainRF():
	passenger_prepared,passenger_labels,test_set,passenger_test = pt.preProcessData()

	sgd_clf = RandomForestClassifier(n_estimators=100)
	passenger_labels = passenger_labels.values.reshape((len(passenger_labels.values),))
	sgd_clf.fit(passenger_prepared, passenger_labels)

	print(cross_val_score(sgd_clf, passenger_prepared, passenger_labels, cv=3, scoring="accuracy"))

	predict_data = sgd_clf.predict(passenger_test)
	PassengerIds = test_set['PassengerId']
	results = pd.Series(predict_data,name="Survived",dtype=np.int32)
	submission = pd.concat([PassengerIds,results],axis = 1)

	dl.saveData(submission,'RFClassifier.csv')

