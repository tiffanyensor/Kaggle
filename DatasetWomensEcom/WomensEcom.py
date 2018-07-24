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


# disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Read in the data
data = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', skiprows=1, names=['RowNumber', 'ClothingID', 'Age', 'Title', 'Review', 'Rating', 'Recommend', 'PosFeedback', 'Division', 'Department', 'Class'])


#----------------------
# EXPLORE THE DATA
#----------------------

# number of rows, number of columns
n_rows = data.shape[0]
n_cols = data.shape[1]


# Explore variable: ClothingID
# ------------------------

data.ClothingID.isnull().sum()                  # no missing values
print("There are ",data.ClothingID.nunique()," unique items.")


# Explore variable: Age
# ------------------------

data.Age.isnull().sum()                         # no missing values
data.Age.mean()                                 # 43.198
data.Age.median()                               # 41.0
data.Age.min()                                  # 18
data.Age.max()                                  # 99


# Explore variable: Title
# ------------------------
data.Title.isnull().sum()                       # 3810 missing values


# Explore variable: Review
# ------------------------
data.Review.isnull().sum()                      # 845 missing values



# Explore variable: Rating
# ------------------------
data.Rating.isnull().sum()                      # no missing values
data.Rating.mean()
data.Rating.median()
data.Rating.value_counts()


# Explore variable: Recommend
# ------------------------
data.Recommend.isnull().sum()                   # no missing values
data.Recommend.value_counts()
data.Recommend.value_counts(1)                  # as percent


# Explore variable: PosFeedback
# ------------------------
data.PosFeedback.isnull().sum()                 # no missing values
data.PosFeedback.value_counts()


# Explore variable: Division
# ------------------------
data.Division.isnull().sum()                    # 14 missing values
data.Division.unique()                          # ['Initmates', 'General', 'General Petite', nan]
data.Division.value_counts()




# Explore variable: Department
# ------------------------
        
data.Department.isnull().sum()                  # 14 missing values
data.Department.unique()                        # ['Intimate', 'Dresses', 'Bottoms', 'Tops', 'Jackets', 'Trend', nan]
data.Department.value_counts()


# Explore variable: Class
# ------------------------
        
data.Class.isnull().sum()                       # 14 missing values
data.Class.unique()  
data.Class.value_counts()            

# Casual bottoms appear only twice - reclassify as Pants


# Chemises appears only once - reclassify as Blouses
        
        
        
        
#----------------------
# MISSING DATA
#----------------------
        
print("The following rows are missing values: ")
print("RowNumber\t ClothingID\t Division\t Department\t Class")
print("----------------------------------------------------------------------")
for i in range(0,n_rows):
    if (pd.isnull(data.Division[i]) or pd.isnull(data.Department[i]) or pd.isnull(data.Class[i])):
        print(data.RowNumber[i],"\t\t",data.ClothingID[i],"\t\t", data.Division[i],"\t\t", data.Department[i],"\t\t", data.Class[i])
        

# Read Titles and Reviews for the missing rows:
# compare to Division/Department/Class for comparable items
# e.g., review for sock corresponds to Division = Initmates, Department = Intimate. Class = Legwear

# ClothingID 72 (row 9444) is for a sock
  # ------------------------------------      
data.Division[9444] = 'Initmates'
data.Department[9444] = 'Intimate'
data.Class[9444] = 'Legwear'
        
# Clothing ID = 492 (rows  13767, 13768, 13787) is a hoodie
# ------------------------------------
data.Division[13767] = 'Initmates'
data.Department[13767] = 'Intimate'
data.Class[13767] = 'Lounge'

data.Division[13768] = 'Initmates'
data.Department[13768] = 'Intimate'
data.Class[13768] = 'Lounge'

data.Division[13787] = 'Initmates'
data.Department[13787] = 'Intimate'
data.Class[13787] = 'Lounge'

# Clothing ID 152 (rows 16216, 16221, 16223) is for leg warmers
# ------------------------------------
data.Division[16216] = 'Initmates'
data.Department[16216] = 'Intimate'
data.Class[16216] = 'Legwear'    

data.Division[16221] = 'Initmates'
data.Department[16221] = 'Intimate'
data.Class[16221] = 'Legwear'  

data.Division[16223] = 'Initmates'
data.Department[16223] = 'Intimate'
data.Class[16223] = 'Legwear'    

# Clothing ID 184 (rows 18626, 18671) is for tights/leggings
# ------------------------------------
data.Division[18626] = 'Initmates'
data.Department[18626] = 'Intimate'
data.Class[18626] = 'Legwear'  

data.Division[18671] = 'Initmates'
data.Department[18671] = 'Intimate'
data.Class[18671] = 'Legwear' 


# Clothing ID 772 (row 20088) is for a sweatshirt
# ------------------------------------
data.Division[20088] = 'General'
data.Department[20088] = 'Tops'
data.Class[20088] = 'Knits'   

           
# Clothing ID 665 (row 21532) is most likely for an undergarment
# ------------------------------------

data.Division[21532] = 'Initmates'
data.Department[21532] = 'Intimate'
data.Class[21532] = 'Intimates'     


# Clothing ID 136 (rows 22997, 23006, 23011) is for socks
# ------------------------------------
                
data.Division[22997] = 'Initmates'
data.Department[22997] = 'Intimate'
data.Class[22997] = 'Legwear'

data.Division[23006] = 'Initmates'
data.Department[23006] = 'Intimate'
data.Class[23006] = 'Legwear'

data.Division[23011] = 'Initmates'
data.Department[23011] = 'Intimate'
data.Class[23011] = 'Legwear'



# verify:
#data.Division.value_counts()
#data.Department.value_counts()
#data.Class.value_counts()




#----------------------
# WORD MAP
#----------------------







