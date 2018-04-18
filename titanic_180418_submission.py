#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 08:26:48 2018

@author: tiffanyensor

Kaggle competition: Titanic



"""

#---------------------
# Import data
#---------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# X variables: pclass, sex, age, subsp, parch, fare, embarked
train_data = pd.read_csv('train.csv')
X_train = train_data.iloc[:,[2,4,5,6,7,9,11]].values
y_train = train_data.iloc[:,1].values

test_data = pd.read_csv('test.csv')
X_test = test_data.iloc[:,[1,3,4,5,6,8,10]].values


#---------------------
# Missing Data
#---------------------


# Replace missing data for "embarked" with the most freq embarked location
def find_most_freq(col):
    count_C = 0
    count_S = 0
    count_Q = 0
    
    for i in range(0,len(col)-1):
        if col[i] == 'Q':
            count_Q = count_Q + 1
        elif col[i] == 'S':
            count_S = count_S + 1
        else:
            count_C = count_C + 1
    
    if max(count_C, count_S, count_Q) == count_C:
        return 'C'
    elif max(count_C, count_S, count_Q) == count_Q:
        return 'Q'
    else:
        return 'S'
            
for i in range(0, train_data.shape[0]-1):
    if pd.isnull(X_train[i,6]):
        X_train[i,6] = find_most_freq(X_train[:,6])


# Replace missing "fare" values in X_test with the mean fare        

from sklearn.preprocessing import Imputer
imputer_fare = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
imputer_fare = imputer_fare.fit(X_test[:,5].reshape(-1,1))
X_test[:,5] = imputer_fare.transform(X_test[:,5].reshape(-1,1)).flatten()

# Missing age data: for an inital attempt, remove "age" as a variable.

X_train = X_train[:,[0,1,3,4,5,6]]
X_test = X_test[:,[0,1,3,4,5,6]]


#---------------------
# Preprocessing
#---------------------

# Encode dummy variables (sex and embarked, index 1 and 5)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_sex = LabelEncoder()
X_train[:,1] = labelencoder_sex.fit_transform(X_train[:,1])
X_test[:,1] = labelencoder_sex.transform(X_test[:,1])

labelencoder_embarked = LabelEncoder()
X_train[:,5] = labelencoder_embarked.fit_transform(X_train[:,5])
X_test[:,5] = labelencoder_embarked.transform(X_test[:,5])

onehotencoder = OneHotEncoder(categorical_features=[1,5])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.transform(X_test).toarray()

# Dummy variable trap
X_train = X_train[:,[1,3,4,5,6,7,8]]
X_test = X_test[:,[1,3,4,5,6,7,8]]


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#---------------------
# Apply Kernel SVM to predict "Survived"
#---------------------

# Best model is Kernel SVM (see below)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv=10)
print("mean: ", round(accuracies.mean(), 2))
print("standard dev: ",round(accuracies.std(),2))

# grid search for parameter tuning

from sklearn.model_selection import GridSearchCV
parameters = [{"C": [155, 156, 157, 158, 159, 160], "kernel":['rbf'], "gamma": [0.03, 0.02, 0.01, 0.009, 0.008]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C=155, gamma = 0.01)
classifier.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train, y=y_train, cv=10)
print("mean: ", round(accuracies.mean(), 2))
print("standard dev: ",round(accuracies.std(),2))


y_pred = classifier.predict(X_test)


#---------------------
# Export Results to CSV
#---------------------

Predicted_results = np.array([test_data.iloc[:,0].values, y_pred]).transpose()
Predicted_results = pd.DataFrame(Predicted_results, columns=['PassengerID', 'Survived'])
Predicted_results.to_csv('Titanic_predictions.csv', index=False)

#---------------------
# COMPARE CLASSIFICATION MODELS
#---------------------

def do_logistic_regression(X,y):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression()
    classifier.fit(X, y)

    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X=X, y=y, cv=10)
    return round(accuracies.mean(), 2), round(accuracies.std(),2)

def do_KNN(X,y):

    from sklearn.neighbors import KNeighborsClassifier    
    classifier = KNeighborsClassifier(n_neighbors=7, metric='minkowski', leaf_size=41, p=2)
    classifier.fit(X, y)
    
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X=X, y=y, cv=10)
    return round(accuracies.mean(), 2), round(accuracies.std(),2)


def do_linear_SVM(X,y):

    from sklearn.svm import SVC
    classifier = SVC(kernel='linear')
    classifier.fit(X, y)

    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X=X, y=y, cv=10)
    return round(accuracies.mean(), 2), round(accuracies.std(),2)    


def do_kernel_SVM(X,y):
    

    from sklearn.svm import SVC
    classifier = SVC(kernel = 'rbf')
    classifier.fit(X, y)

    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X=X, y=y, cv=10)
    return round(accuracies.mean(), 2), round(accuracies.std(),2)


def do_naive_bayes(X,y):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(X, y)
    
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X=X, y=y, cv=10)
    return round(accuracies.mean(), 2), round(accuracies.std(),2)

def do_decision_tree(X,y):
    
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(criterion='entropy')
    classifier.fit(X, y)

    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X=X, y=y, cv=10)
    return round(accuracies.mean(), 2), round(accuracies.std(),2)


def do_random_forest(X,y):

    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy')
    classifier.fit(X, y)
   
    from sklearn.model_selection import cross_val_score
    accuracies = cross_val_score(estimator = classifier, X=X, y=y, cv=10)
    return round(accuracies.mean(), 2), round(accuracies.std(),2)

    
def model_comparison(X_train,y_train):
    print('logistic regression: \t', do_logistic_regression(X_train,y_train))
    print('K-NN: \t\t\t', do_KNN(X_train,y_train))
    print('linear SVM: \t\t', do_linear_SVM(X_train,y_train))
    print('kernel SVM: \t\t', do_kernel_SVM(X_train,y_train))
    print('Naive bayes: \t\t', do_naive_bayes(X_train,y_train))    
    print('decision tree: \t\t', do_decision_tree(X_train,y_train))
    print('random forest: \t\t', do_random_forest(X_train,y_train))
