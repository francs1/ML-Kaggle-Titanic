# trainingModel.py
import sys

import pandas as pd
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

from sklearn import linear_model

from constantInit import *
import dataLoad as dl

import preTreatment as pt

np.random.seed(42)


def trainLR():
	passenger_prepared,passenger_labels,test_set,passenger_test = pt.preProcessData()

	lr_clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6, solver='lbfgs')
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