import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatize
from sklearn.feature_extraction.text import CountVectorizer


# %matplotlib inline
# sns.set_style('whitegrid')
# plt.style.use('fivethirtyeight')

# example text for model training
simple_train=['call you tonight','call me a cab','please call me..... please']

# Fitting the vectorizer on the training data learns the vocabulary from the text.
vect=CountVectorizer()
vect.fit(simple_train)
#After fitting the vectorizer, we can inspect the vocabulary learned from the text data. 
# This returns all the unique words found in the training data.
# print(vect.get_feature_names_out())

#transform the training data into 'document-term matrix'
# This matrix represents the frequency of each word in each document.
simple_train_dtm=vect.transform(simple_train)
#converting sparse matrix into a dense matrix
# print(simple_train_dtm.toarray())

# Each row of the DataFrame corresponds to a text, and each column corresponds to a word in the vocabulary. 
# The cell values represent the frequency of each word in each document.
df=pd.DataFrame(simple_train_dtm.toarray(),columns=vect.get_feature_names_out())
# print(df)

#testing this vectorizer using an example training set, if its working properly or no
simple_test=['please call me tonight!']
simple_test_dtm=vect.transform(simple_test)
#ok identifying the words properly
# print(pd.DataFrame(simple_test_dtm.toarray(),columns=vect.get_feature_names_out()))