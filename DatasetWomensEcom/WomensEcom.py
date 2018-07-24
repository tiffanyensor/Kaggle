#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 08:55:27 2018

@author: tiffanyensor
"""

#----------------------
# Womens Ecom Dataset
#----------------------

# Imports

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Read in the data
data = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', skiprows=1, names=['RowNumber', 'ClothingID', 'Age', 'Title', 'Review', 'Rating', 'Recommend', 'PosFeedback', 'Division', 'Department', 'Class'])


#----------------------
# Explore the Data
#----------------------

# number of rows, number of columns
n_rows = data.shape[0]
n_cols = data.shape[1]


# Explore variable: ClothingID
# ------------------------

data.ClothingID.isnull().sum()    # no missing values
print("There are ",data.ClothingID.nunique()," unique items.")


# Explore variable: Age
# ------------------------

data.Age.isnull().sum()    # no missing values
print("Mean age = ",data.Age.mean())
print("Median age = ",data.Age.median())
print("Min age = ",data.Age.min())
print("Max age = ",data.Age.max())


# Explore variable: Title
# ------------------------
print("There are ",data.Title.isnull().sum()," reviews missing titles.")


# Explore variable: Review
# ------------------------
print("There are ",data.Review.isnull().sum()," missing reviews.")



# Explore variable: Rating
# ------------------------
data.Rating.isnull().sum()    # no missing values
data.Rating.mean()
data.Rating.median()
data.Rating.value_counts()


# Explore variable: Recommend
# ------------------------
data.Recommend.isnull().sum()    # no missing values
data.Recommend.value_counts()
data.Recommend.value_counts(1)   # as percent


# Explore variable: PosFeedback
# ------------------------
data.PosFeedback.isnull().sum()    # no missing values
data.PosFeedback.value_counts()


# Explore variable: Division
# ------------------------
print("There are ",data.Division.isnull().sum()," missing values for Division.")

data.Division.unique()

# Explore variable: Department
# ------------------------

data.Department.unique()


# Explore variable: Class
# ------------------------
data.Class.unique()
