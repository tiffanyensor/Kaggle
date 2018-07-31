#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 08:55:27 2018

@author: tiffanyensor
"""

# TO DO LIST:
# TRY: making new features: e.g., combine all "tops", "blouses", "sweaters", etc. to a single new feature
# Combine the 2 plot types into a single figure for easier comparability
# MAKE a data frame containing the results: e.g., word, indexmap_index, n_occurance, n_pos_occ, n_neg_occ, % pos reviews, % neg revies, etc. 
# Correct for more positive than negative reviews... need to either subsample pos or extrapolate neg or divide by n_words in rev
# Better to run neutral reviews through? or just pos/neg
# incorporate title into review
# Add percents to histogram figure
# CAP curve for reccommend

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Womens Ecom Dataset
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# disable SettingWithCopyWarning
pd.options.mode.chained_assignment = None

# Read in the data
data = pd.read_csv('Womens Clothing E-Commerce Reviews.csv', skiprows=1, names=['RowNumber', 'ClothingID', 'Age', 'Title', 'Review', 'Rating', 'Recommend', 'PosFeedback', 'Division', 'Department', 'Class'])


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# SOME PLOT FUNCTIONS
##------------------------------------------------------------------------------
#------------------------------------------------------------------------------


def histogram_by_rating(binned_data, rating, var_name):
    
    d = {'binned_data': binned_data, 'rating': rating.astype(str)}
    df = pd.DataFrame(d)
    
    # group data by rating and binned_data
    data_by_rating = df.groupby(['binned_data', 'rating'])['binned_data'].count().unstack()
    data_by_rating = data_by_rating.fillna(0)
    bin_names=data_by_rating.index
   
    # set up plot area, leaving room for legend on the right
    fig = plt.figure()
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # Stacked bar plots
    p1=ax.bar(bin_names, data_by_rating['1'], color='red')
    p2=ax.bar(bin_names, data_by_rating['2'], bottom=data_by_rating['1'], color='gold')
    p3=ax.bar(bin_names, data_by_rating['3'], bottom=data_by_rating['2'], color='yellow')
    p4=ax.bar(bin_names, data_by_rating['4'], bottom=data_by_rating['3'], color='palegreen')
    p5=ax.bar(bin_names, data_by_rating['5'], bottom=data_by_rating['4'], color='mediumseagreen')

    # Plot specifications and legend
    ax.legend((p1, p2, p3, p4, p5), ('Rating = 1', 'Rating = 2', 'Rating = 3', 'Rating = 4', 'Rating = 5'), fontsize=11, loc = "upper left", bbox_to_anchor = (1, 1))
    plt.title("Histogram of "+var_name, fontsize=16)
    plt.xlabel(var_name, fontsize=14)
    plt.xticks(rotation=90)
    plt.ylabel("Count", fontsize=14)
    plt.savefig(var_name+"Histogram.pdf", bbox_inches="tight")
    plt.show()

    
def histogram_by_rating_by_percent(binned_data, rating, var_name):
    
    # convert to dataframe
    d = {'binned_data': binned_data, 'rating': rating.astype(str)}
    df = pd.DataFrame(d)

    # group data by rating and binned_data
    data_by_rating = df.groupby(['binned_data', 'rating'])['binned_data'].count().unstack()
    data_by_rating = data_by_rating.fillna(0)
    bin_names=data_by_rating.index
    counts = data_by_rating.sum(axis=1)    # get sum across row, ie, sum for each group/category
    
    # set up plot area, leaving room for legend on the right
    fig = plt.figure()
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    
    # stacked barplots
    p1 = ax.bar(bin_names, data_by_rating['1']/counts, color='red')
    p2 = ax.bar(bin_names, data_by_rating['2']/counts, bottom=data_by_rating['1']/counts, color='gold')
    p3 = ax.bar(bin_names, data_by_rating['3']/counts, bottom=data_by_rating['1']/counts + data_by_rating['2']/counts, color='yellow')
    p4 = ax.bar(bin_names, data_by_rating['4']/counts, bottom=data_by_rating['1']/counts + data_by_rating['2']/counts + data_by_rating['3']/counts, color='palegreen')
    p5 = ax.bar(bin_names, data_by_rating['5']/counts, bottom=data_by_rating['1']/counts + data_by_rating['2']/counts + data_by_rating['3']/counts + data_by_rating['4']/counts, color='mediumseagreen')

    # Legend and plot specifications
    ax.legend((p1, p2, p3, p4, p5), ('Rating = 1', 'Rating = 2', 'Rating = 3', 'Rating = 4', 'Rating = 5'), fontsize=12,loc = "upper left", bbox_to_anchor = (1, 1))
    plt.xlabel(var_name, fontsize=14)
    plt.xticks(rotation=90)
    plt.ylabel("Percent", fontsize=14)
    plt.title(var_name+" by percent", fontsize=16)
    plt.savefig(var_name+"ByPercent.pdf", bbox_inches="tight")
    plt.show() 
    

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------    
# EXPLORE THE DATA
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

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

labels=['(10-20]', '(20-30]', '(30-40]', '(40-50]', '(50-60]', '(60-70]', '(70-80]', '(80-90]', '(90-100]']
data['BinnedAge'] = pd.cut(data.Age, bins=[10,20,30,40,50,60,70,80,90,100], labels=labels)

histogram_by_rating(data.BinnedAge, data.Rating, 'Age')  
histogram_by_rating_by_percent(data.BinnedAge, data.Rating, 'Age')  


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

histogram_by_rating(data.Division, data.Rating, 'Division')  
histogram_by_rating_by_percent(data.Division, data.Rating, 'Division')  


# Explore variable: Department
# ------------------------
        
data.Department.isnull().sum()                  # 14 missing values
data.Department.unique()                        # ['Intimate', 'Dresses', 'Bottoms', 'Tops', 'Jackets', 'Trend', nan]
data.Department.value_counts()

histogram_by_rating(data.Department, data.Rating, 'Department')    
histogram_by_rating_by_percent(data.Department, data.Rating, 'Department')  


# Explore variable: Class
# ------------------------
        
data.Class.isnull().sum()                       # 14 missing values
data.Class.unique()  
data.Class.value_counts()   

histogram_by_rating(data.Class, data.Rating, 'Class')    
histogram_by_rating_by_percent(data.Class, data.Rating, 'Class')    

# ratings: need to replace NAN with zero

# Casual bottoms appear only twice - reclassify as Pants?
# Chemises appears only once - reclassify as Blouses?
        
        
        
#------------------------------------------------------------------------------        
#------------------------------------------------------------------------------
# REPLACE MISSING DATA
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------ 
       
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


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# PREPARE REVIEW DATA FOR NLP
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

reviews = []
label = []

# ignore missing review text data
for i in range(0, len(data.Review)):
    if pd.isnull(data.Review[i]) == False:
        reviews.append(data.Review[i])
        label.append(data.Rating[i])


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# SENTIMENT ANALYSIS
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Tokenizer function
def my_tokenizer(review):
    review = review.lower()                                 # convert to lowercase
    tokens = review.split()                                 # split review into words
    tokens = [t for t in tokens if t.isalpha()]             # remove non-alpha characters                               # convert to lowercase
    tokens = [lemmatizer.lemmatize(t) for t in tokens]      # lemmatize
    tokens = [t for t in tokens if len(t) > 2]              # remove short words
    tokens = [t for t in tokens if t not in stop_words]     # remove stopwords
    return tokens


corpus = []                 # list of tokenized review strings
tokenized_words = []        # list containing all words found in corpus (repeats included)
word_index_map = {}         # contains unique words and their (most recent) correpsonding position in tokenized_words list
current_index=0             

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

for i in range(0, len(reviews)):
    text_review = " "
    review = reviews[i]
    review = my_tokenizer(review)
#    review = [stemmer.stem(word) for word in review] 
    
    # cyle through all the words in the tokenized reviews
    for word in review:
        current_word_count = 1
        tokenized_words.append(word)
        text_review = text_review + " " + word
        if word not in word_index_map:
            word_index_map[word]=current_index
            current_index += 1
    corpus.append(text_review)
    
print("There are ",len(set(tokenized_words))," unique tokens.")

from collections import Counter
word_count = Counter(tokenized_words)
#word_count = Counter(tokenized_words).most_common()      # desc order

# COUNT VECTORIZER
# -----------------------

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(min_df=10)    # words should occur in at least __ reviews

# build sparse matrix from the corpus
X = tfidf.fit_transform(corpus)
vocab = tfidf.vocabulary_

feature_names = tfidf.get_feature_names()    # same entries as word-index-map
idf = tfidf.idf_
tfidf_scores = dict(zip(feature_names, idf))

# store results in a dataframe
results = pd.DataFrame(feature_names, columns=['Word'])
results['TFIDF_Score']=idf
results['Total_Count'] = [word_count[word] for word in results['Word']]
results['Word_Map_Index'] = [word_index_map[word] for word in results['Word']]


def get_top_tfidf_words(word, score, top_n=25):
    df = pd.DataFrame(feature_names, columns=['Word'])
    df['TFIDF_Score']=idf
    sorted=df.sort_values(by='TFIDF_Score', ascending=False)
    top = sorted[0:top_n]
    return top

print("Top TFIDF Words:")
print(get_top_tfidf_words(feature_names, idf))


# Assign a value to y based on positive (rating = 4, 5), negative (rating = 1,2), or neutral (rating=3) review
y = [0]*len(label)
for index, val in enumerate(label):
    if val==1 or val==2:                # negative review
        y[index] = -1
    elif val == 3:                      # neutral review
        y[index] = 0
    else:
        y[index] = 1                    # positive review



# APPLY ML ALGORITHM
# -----------------------


# TRY LOGISTIC REGRESSION
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)

print("Confusion matrix for logistic regression : ")
print(cm)
print("Accuracy for logistic regression = ",accuracy_score(y_test, y_pred))

# Calculate weights (pos/neg) associated with each word
weights = classifier.coef_[0].tolist()
results["LogReg_Weights"] = weights
results["Abs_LogReg_Weights"] = [abs(number) for number in weights]

top_n= 30

most_neg =results.nlargest(top_n, 'LogReg_Weights').iloc[:,[0,4]]
plt.figure(1, figsize = (4,10))
plt.barh(most_neg["Word"], most_neg["LogReg_Weights"])
plt.gca().invert_yaxis()
plt.xlim(0.9*min(most_neg["LogReg_Weights"]), 1.01*max(most_neg["LogReg_Weights"]))
plt.title("Top "+str(top_n)+" Most Negative Words", fontsize=16)
plt.xlabel("Magnitude of Logistic Regression Weight")
plt.savefig("MostNegativeWords.pdf", bbox_inches="tight")
plt.show() 


#print(results.nlargest(top_n, 'LogReg_Weights').iloc[:,[0,4]])


most_pos = results.nsmallest(top_n, 'LogReg_Weights').iloc[:,[0,4,5]]
plt.figure(1, figsize = (4,10))
plt.barh(most_pos["Word"], most_pos["Abs_LogReg_Weights"])
plt.xlabel("Magnitude of Logistic Regression Weight")
plt.gca().invert_yaxis()
plt.xlim(0.9*min(most_pos["Abs_LogReg_Weights"]), 1.01*max(most_pos["Abs_LogReg_Weights"]))
#plt.xticks(rotation=90)
plt.title("Top "+str(top_n)+" Most Positive Words", fontsize=16)
plt.savefig("MostPositiveWords.pdf", bbox_inches="tight")
plt.show() 


print("MOST NEUTRAL WORDS:")
print(results.nsmallest(top_n, 'Abs_LogReg_Weights').iloc[:,[0,4]])



"""
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# WORD CLOUD
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
      
# Separate into positive ratings (5 and 4) and negative reviews (1 and 2)
pos_reviews = []
neg_reviews = []

for i in range(len(reviews)):
    if (y[i] == -1):
        neg_reviews.append(reviews[i])          # 2370 neg reviews
    if (y[i] == 1):
        pos_reviews.append(reviews[i])          # 17448 pos reviews 

       
neg_word_string = ''
neg_tokens = []
for rev in neg_reviews:
    tokens = my_tokenizer(rev)
    for word in tokens:
        neg_word_string = neg_word_string + ' ' + word
        neg_tokens.append(word)

neg_word_count = Counter(neg_tokens).most_common()
        
   
from wordcloud import WordCloud

# The words "fit", "dress", "top", "fabric" appear in both, so remove them
wordcloud = WordCloud(height=500, width=500, background_color="white", colormap="inferno", max_words=100, stopwords=['fit', 'dress', 'top', 'fabric']).generate(neg_word_string)
plt.figure(figsize = (7,7))
plt.imshow(wordcloud)
plt.title('Word Cloud for Negative Reviews', fontsize=18)
plt.axis('off')
plt.savefig("NegativeWordCloud.pdf")
plt.show()

"""

"""
pos_word_string = ''
pos_tokens = []
for rev in pos_reviews:
    tokens = my_tokenizer(rev)
    for word in tokens:
        pos_word_string = pos_word_string + ' ' + word
        pos_tokens.append(word)

pos_word_count = Counter(pos_tokens).most_common()

wordcloud = WordCloud(height=500, width=500, background_color="white", max_words = 100, stopwords=['fit', 'dress', 'top', 'fabric']).generate(pos_word_string)
plt.figure(figsize = (7,7))
plt.imshow(wordcloud)
plt.title('Word Cloud for Positive Reviews', fontsize=18)
plt.axis('off')
plt.savefig("PositiveWordCloud.pdf")
plt.show()
"""
