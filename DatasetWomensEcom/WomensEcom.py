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
data.ClothingID.nunique()                       # 



# Explore variable: Age
# ------------------------

data.Age.isnull().sum()                         # no missing values
data.Age.mean()                                 # 43.198
data.Age.median()                               # 41.0
data.Age.min()                                  # 18
data.Age.max()                                  # 99

data['BinnedAge'] = pd.cut(data.Age, bins=[10,20,30,40,50,60,70,80,90,100])

counts_by_rating = data.groupby(by=['Rating']).count()
age_by_rating = data.Age.groupby(by=[data.Rating]).count()

plt.hist(data.Age, bins=[10,20,30,40,50,60,70,80,90,100])
plt.xlabel("Age", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Hisotgram of Age Distribution", fontsize=16)
plt.savefig("AgeHistogram.pdf")
plt.show()



# Explore variable: Title
# ------------------------
data.Title.isnull().sum()                       # 3810 missing values


# Explore variable: Review
# ------------------------
data.Review.isnull().sum()                      # 845 missing values (22641 total reviews)



# Explore variable: Rating
# ------------------------
data.Rating.isnull().sum()                      # no missing values
data.Rating.mean()                              # 4.196
data.Rating.median()                            # 5.0
data.Rating.value_counts()

xdata = data.Rating.value_counts().keys().tolist()
ydata = data.Rating.value_counts().tolist()
plt.bar(data.Rating.value_counts().keys().tolist(), data.Rating.value_counts().tolist())
plt.title("Rating Histogram", fontsize=16)
plt.xlabel("Rating")
plt.ylabel("Count")
for i in range(0,5):
    plt.text(xdata[i]-0.2, ydata[i]+100, str(ydata[i]))
plt.savefig("RatingHistogram.pdf")
plt.show()


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

# Casual bottoms appear only twice - reclassify as Pants?


# Chemises appears only once - reclassify as Blouses?
        
        
        
        
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
# BUILD DATASET
#----------------------


reviews = []
label = []

# ignore missing review text data
for i in range(0, len(data.Review)):
    if pd.isnull(data.Review[i]) == False:
        reviews.append(data.Review[i])
        label.append(data.Rating[i])


#----------------------
# SENTIMENT ANALYSIS
#----------------------


import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# tokenizer function
def my_tokenizer(review):
    review = review.lower()                                 # convert to lowercase
    tokens = review.split()                                 # split review into words
    tokens = [t for t in tokens if t.isalpha()]             # remove non-alpha characters                               # convert to lowercase
    tokens = [lemmatizer.lemmatize(t) for t in tokens]      # lemmatize
    tokens = [t for t in tokens if len(t) > 2]              # remove short words
    tokens = [t for t in tokens if t not in stop_words]     # remove stopwords
    return tokens

corpus = []
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
for i in range(0, len(reviews)):
    tokens = my_tokenizer(reviews[i]) 
    text_review = " "
    review = reviews[i]
    review = my_tokenizer(review)
    review = [stemmer.stem(word) for word in review]
    for word in review:
        text_review = text_review + " " + word
    corpus.append(text_review)
    

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=10000)

X = cv.fit_transform(corpus).toarray()

# pos has rating 4 or 5, neutral has 3, neg has 1 or 2

y = [0]*len(label)
for index, val in enumerate(label):
    if val==1 or val==2:                # negative review
        y[index] = -1
    elif val == 3:                      # neutral review
        y[index] = 0
    else:
        y[index] = 1                    # positive review


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)

print("Confusion matrix: ")
print(cm)
print("Accuracy = ",accuracy_score(y_test, y_pred))

      
# Separate into positive ratings (5 and 4) and negative reviews (1 and 2)
pos_reviews = []
neg_reviews = []

for i in range(len(reviews)):
    if (y[i] == -1):
        neg_reviews.append(reviews[i])          # 2370 neg reviews
    if (y[i] == 1):
        pos_reviews.append(reviews[i])          # 17448 pos reviews



#----------------------
# WORD CLOUD
#----------------------
      
        
neg_word_string = ''
for rev in neg_reviews:
    tokens = my_tokenizer(rev)
    for word in tokens:
        neg_word_string = neg_word_string + ' ' + word
   
from wordcloud import WordCloud

# The words "fit", "dress", "top", "fabric" appear in both, so remove them

wordcloud = WordCloud(height=500, width=500, background_color="white", colormap="inferno", max_words=100, stopwords=['fit', 'dress', 'top', 'fabric']).generate(neg_word_string)
plt.figure(figsize = (7,7))
plt.imshow(wordcloud)
plt.title('Word Cloud for Negative Reviews', fontsize=18)
plt.axis('off')
plt.show()

"""
pos_word_string = ''
for rev in pos_reviews:
    tokens = my_tokenizer(rev)
    for word in tokens:
        pos_word_string = pos_word_string + ' ' + word

wordcloud = WordCloud(height=500, width=500, background_color="white", max_words = 100, stopwords=['fit', 'dress', 'top', 'fabric']).generate(pos_word_string)
plt.figure(figsize = (7,7))
plt.imshow(wordcloud)
plt.title('Word Cloud for Positive Reviews', fontsize=18)
plt.axis('off')
plt.show()
"""


