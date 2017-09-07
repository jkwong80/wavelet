
import os, sys, time
import matplotlib.pyplot as plt

import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_curve, roc_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

number_datapoints = 100000


X,y = make_classification(n_samples = number_datapoints, n_features = 10, n_redundant = 0, n_informative = 10, random_state = 1, n_classes = 3, weights = [.7 , .2, .1])


# Binarize the output
y = label_binarize(y, classes=list(set(y)))
n_classes = y.shape[1]


# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)

train_mask = np.ones(len(y)) == 0
train_mask[0:(number_datapoints/2)] = True

clf = RandomForestClassifier(n_estimators=100, n_jobs=4, class_weight=[{0:100,1:1}, {0:1,1:1}, {0:1,1:1}])

clf.fit(X_train, y_train)

prob = clf.predict_proba(X)

prob_test = clf.predict_proba(X_test)

prediction_test = clf.predict(X_test)

plt.figure()
plt.grid()
for i in xrange(n_classes):
    fpr, tpr, threshold = roc_curve(y_test[:,i], prob_test[i][:,1])
    plt.plot(fpr, tpr, '.', label = 'class: {}'.format(i))

plt.legend()
plt.xlabel('fpr')
plt.ylabel('tpr')


plt.figure()
plt.grid()
for i in xrange(n_classes):
    precision, recall, threshold = precision_recall_curve(y_test[:,i], prob_test[i][:,1])
    plt.plot(precision, recall, '.', label = 'class: {}'.format(i))

plt.legend()
plt.xlabel('precision')
plt.ylabel('recall')
