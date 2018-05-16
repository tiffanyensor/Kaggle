#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 16:59:03 2018

@author: tiffanyensor
"""

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

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

#---------------------
# New variable: FamilySize
#---------------------

# combine sibsp and parch into a new variable, FamilySize
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch']

#---------------------
# New variable: Title
#---------------------

# get full list of titles:
from collections import Counter
title_list = train_data['Name'].str.extract(',\s(\w+)\.')
title_count=Counter(title_list)

title_train=[]
for name in train_data['Name']:
    if ('Mrs.' in name) or ('Mme.' in name):
        title_train.append('Mrs')
    elif ('Mr.' in name):
        title_train.append('Mr')
    elif ('Miss.' in name) or ('Mlle.' in name):
        title_train.append('Miss')
    elif 'Master.' in name:
        title_train.append('Master')
    else:
        title_train.append('Other')
train_data['Title'] = title_train

title_test=[]
for name in test_data['Name']:
    if ('Mrs.' in name) or ('Mme.' in name):
        title_test.append('Mrs')
    elif ('Mr.' in name):
        title_test.append('Mr')
    elif ('Miss.' in name) or ('Mlle.' in name):
        title_test.append('Miss')
    elif 'Master.' in name:
        title_test.append('Master')
    else:
        title_test.append('Other')
test_data['Title'] = title_test

#---------------------
# Encode Gender: IsMale
#---------------------

def check_male(s):
    if s=='male':
        return 1
    else:
        return 0
    
train_data['IsMale'] = train_data['Sex'].apply(check_male)
test_data['IsMale'] = test_data['Sex'].apply(check_male)


#---------------------
# Missing Data - Embarked
#---------------------

# There are missing Emmarked values in the train set

for val in train_data['Embarked']:
    if pd.isnull(val):
        print("Embarked value is null")

# Find the most common Embarked value:

emb_count = train_data['Embarked'].value_counts()
emb_max_val = train_data['Embarked'].value_counts().idxmax()
print("The most common embarked value is ", emb_max_val)

# Replace missing Embarked values with the most common value

train_data['Embarked']=train_data['Embarked'].fillna(emb_max_val)


#---------------------
# Missing Data - Fare
#---------------------

# Compare train and test set means for Fare

np.mean(test_data["Fare"])
np.mean(train_data["Fare"])

# Replace missing fare values with the mean

for val in test_data['Fare']:
    if pd.isnull(val):
        print("Fare value is null")

test_data['Fare'] = test_data['Fare'].fillna(np.mean(test_data['Fare']))


#---------------------
# Prepare Train set and Test Set
#---------------------

# Use the following variables: embarked, title, pclass, fare, familysize, title, ismale
# Put the variables that need to be encoded first (embarked, title)

X_train = train_data.iloc[:,[11,13,2,9,12,14]].values
Y_train = train_data.iloc[:,1].values
X_test = test_data.iloc[:,[10,12,1,8,11,13]].values


#---------------------
# Preprocessing - Encode Dummies
#---------------------

# Encode dummy variables Embarked (ind = 0) and Title (ind = 1)
# Note: Sex is already encoded as IsMale

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_embarked = LabelEncoder()
X_train[:,0] = labelencoder_embarked.fit_transform(X_train[:,0])
X_test[:,0] = labelencoder_embarked.transform(X_test[:,0])

labelencoder_title = LabelEncoder()
X_train[:,1] = labelencoder_title.fit_transform(X_train[:,1])
X_test[:,1] = labelencoder_title.transform(X_test[:,1])

onehotencoder = OneHotEncoder(categorical_features=[0,1])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_test = onehotencoder.transform(X_test).toarray()

# Dummy Variable Trap

X_train = X_train[:,[1,2,4,6,7,8,9,10,11]]
X_test = X_test[:,[1,2,4,6,7,8,9,10,11]]

#---------------------
# Preprocessing - Feature Scaling
#---------------------

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


#---------------------
# Compare Classification Models
#---------------------

# See end of script
model_comparison(X_train, Y_train)


#---------------------
# Apply Kernel SVM to predict "Survived"
#---------------------

# Apple Kernel SVM with k-fold cross validation

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf')
classifier.fit(X_train, Y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X=X_train, y=Y_train, cv=10)
print("mean: ", round(accuracies.mean(), 2))
print("standard dev: ",round(accuracies.std(),2))

# Grid search for parameter tuning

from sklearn.model_selection import GridSearchCV
parameters = [{"C": [1,2,3,4, 5, 6, 7], "kernel":['rbf'], "gamma": [0.09, 0.08, 0.075, 0.07, 0.06, 0.05, 0.04]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

# Refit the model with the new parameters

classifier = SVC(kernel = 'rbf', C=2, gamma = 0.075)
classifier.fit(X_train, Y_train)

accuracies = cross_val_score(estimator = classifier, X=X_train, y=Y_train, cv=10)
print("mean: ", round(accuracies.mean(), 2))
print("standard dev: ",round(accuracies.std(),2))

# predicted Survived values:

y_pred = classifier.predict(X_test)

#---------------------
# Export Results to CSV
#---------------------

Predicted_results = np.array([test_data.iloc[:,0].values, y_pred]).transpose()
Predicted_results = pd.DataFrame(Predicted_results, columns=['PassengerID', 'Survived'])
Predicted_results.to_csv('Titanic_predictions_new.csv', index=False)





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
