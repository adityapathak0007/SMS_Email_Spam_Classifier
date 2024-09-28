import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import nltk
import seaborn as sns
from nltk.corpus import stopwords
stopwords.words('english')
import string
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from wordcloud import WordCloud
from collections import Counter
import warnings
# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
import pickle
#string.punctuation
nltk.download('punkt')

# Tried using a different encoding
df = pd.read_csv("D:\\Aditya's Notes\\All Projects\\Sms Spam Detection\\spam.csv", encoding='latin1')
#print(df.sample(5))
#print(df.shape)

#Steps to Follow:
#1. Data Cleaning
#2. EDA
#3. Text Preprocessing
#4. Model Building
#5. Evaluation
#6. Improvement
#7. Website
#8. Deploy

#1. Data Cleaning
#print(df.info()) #for getting the information of data

#we will drop last 3 columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
print(df.info())
#print(df.sample(5))

#Ranaming the cols
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)
print(df.sample(5))

#encoding
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])
print(df['target'])
print(df.head())

#check for missing values
print(df.isnull().sum()) #no null values

#check for duplicate values
print(df.duplicated().sum())

#remove duplicates values
df = df.drop_duplicates(keep='first')
print(df.duplicated().sum())
print(df.shape)

#2. EDA
print(df.head())
print(df['target'].value_counts())

#plt.pie(df['target'].value_counts(), labels=['ham', 'spam'], autopct="%0.2f")
#plt.show()
#as a result we know that our data is imbalanced

#creating three cols for deeper analysis 1. no. of characters, 2. no. of words, 3. no. of sentences in sms
#1. no. of characters
#print(df['text'].apply(len))
df['num_characters'] = df['text'].apply(len)
print(df.head())

#2. no. of words
print(df['text'].apply(lambda x: len(nltk.word_tokenize(x))))
df['num_words'] = df['text'].apply(lambda x: len(nltk.word_tokenize(x)))
print(df.head())

#3. no. of sentences
print(df['text'].apply(lambda x: len(nltk.sent_tokenize(x))))
df['num_sentences'] = df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))
print(df.head())

#for describing
print(df[['num_characters', 'num_words', 'num_sentences']].describe())

#analyze ham and spam separately
#ham
print(df[df['target'] == 0][['num_characters', 'num_words', 'num_sentences']].describe())
#spam
print(df[df['target'] == 1][['num_characters', 'num_words', 'num_sentences']].describe())

'''
# Plotting the histogram for number of characters
sns.histplot(df[df['target'] == 0]['num_characters'], kde=False, label='Ham', color='blue')
sns.histplot(df[df['target'] == 1]['num_characters'], kde=False, label='Spam', color='orange')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Characters for Ham and Spam')
plt.legend()  # Show legend to distinguish between Ham and Spam
plt.show()

# Plotting the histogram for number of words
sns.histplot(df[df['target'] == 0]['num_words'], kde=False, label='Ham', color='blue')
sns.histplot(df[df['target'] == 1]['num_words'], kde=False, label='Spam', color='orange')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.title('Distribution of Number of Words for Ham and Spam')
plt.legend()  # Show legend to distinguish between Ham and Spam
plt.show()

#to see the relations
sns.pairplot(df, hue='target')
plt.show()
'''
#to see the correlations
# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Compute correlation
corr_matrix = numeric_df.corr()
print(corr_matrix)
'''
# Plotting the heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numeric Features')
plt.show()
'''
#3. Text Preprocessing
'''
1. Lower Case
2. Tokenization
3. Removing Special Characters
4. Removing Stopwords and Punctuation
5. Stemming
'''
#creating the fuction which will do all the steps

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


df['transformed_text'] = df['text'].apply(transform_text)
print(df.head())

#Creating WordCloud Object
wc = WordCloud(width=1000, height=1000, min_font_size=10, background_color='white')
# Generating the word cloud from the 'transformed_text' of spam messages
spam_wc = wc.generate(df[df['target'] == 1]['transformed_text'].str.cat(sep=" "))

# Plotting the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')  # Hide the axes
plt.title('Word Cloud for Spam Messages')
#plt.show()

# Generating the word cloud from the 'transformed_text' of ham messages
spam_wc = wc.generate(df[df['target'] == 0]['transformed_text'].str.cat(sep=" "))

# Plotting the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')  # Hide the axes
plt.title('Word Cloud for Ham Messages')
#plt.show()


#for getting spam messeges words
spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
print(len(spam_corpus))

# Get the 30 most common words and their counts
most_common_words = Counter(spam_corpus).most_common(30)

# Create a DataFrame from the most common words
df_most_common = pd.DataFrame(most_common_words, columns=['Word', 'Count'])

# Plotting the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Word', y='Count', data=df_most_common, palette='viridis')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 30 Most Common Words in Spam Messages')
#plt.show()


#for getting ham messeges words
ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)
print(len(ham_corpus))

# Get the 30 most common words and their counts
most_common_words1 = Counter(ham_corpus).most_common(30)

# Create a DataFrame from the most common words
df_most_common1 = pd.DataFrame(most_common_words1, columns=['Word', 'Count'])

# Plotting the bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Word', y='Count', data=df_most_common1, palette='viridis')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 30 Most Common Words in Ham Messages')
#plt.show()


#4. Model Building

#X = cv.fit_transform(df['transformed_text']).toarray()
#using tfidf vectorizer
X = tfidf.fit_transform(df['transformed_text']).toarray()
print(X.shape)

y = df['target'].values
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Instantiate the classifiers
gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()

# Fit and predict with Gaussian Naive Bayes
gnb.fit(X_train, y_train)
y_pred1 = gnb.predict(X_test)

print("GaussianNB Accuracy:", accuracy_score(y_test, y_pred1))
print("GaussianNB Confusion Matrix:\n", confusion_matrix(y_test, y_pred1))
print("GaussianNB Precision Score:", precision_score(y_test, y_pred1))

# Fit and predict with Multinomial Naive Bayes
mnb.fit(X_train, y_train)
y_pred2 = mnb.predict(X_test)

print("MultinomialNB Accuracy:", accuracy_score(y_test, y_pred2))
print("MultinomialNB Confusion Matrix:\n", confusion_matrix(y_test, y_pred2))
print("MultinomialNB Precision Score:", precision_score(y_test, y_pred2))

# Fit and predict with Multinomial Naive Bayes
bnb.fit(X_train, y_train)
y_pred3 = bnb.predict(X_test)

print("BernoulliNB Accuracy:", accuracy_score(y_test, y_pred3))
print("BernoulliNB Confusion Matrix:\n", confusion_matrix(y_test, y_pred3))
print("BernoulliNB Precision Score:", precision_score(y_test, y_pred3))

#we are choosing tfidf --> mnb our best performing model


'''
#we are testing different models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

#Creating Objects
svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)

clfs = {
    'SVC' : svc,
    'KNC' : knc,
    'MNB' : mnb,
    'DTC' : dtc,
    'LRC' : lrc,
    'RFC' : rfc,
    #'abc' : abc,
    #'BC' : bc,
    'ETC' : etc,
    #'GBDT' : gbdt,
    #'xgb' : xgb
}
X = tfidf.fit_transform(df['transformed_text']).toarray()

#X = scalar.fit_transform(x) not used because performance decreased

#appending the num_character col to X
#X = np.hstack((X,df['num_characters'].values.reshape(-1,1)))

y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
def train_classifier(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    return accuracy,precision

accuracy_scores = []
precision_scores = []

for name, clf in clfs.items():
    current_accuracy, current_precision = train_classifier(clf, X_train, y_train, X_test, y_test)

    print("For ", name)
    print("Accuracy - ", current_accuracy)
    print("Precision - ", current_precision)

    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)

performance_df = pd.DataFrame({'Algorithm' : clfs.keys(), 'Accuracy' : accuracy_scores, 'Precision' : precision_scores}).sort_values('Precision', ascending=False)

print(performance_df)

#Plotting Graph
performance_df = pd.melt(performance_df, id_vars="Algorithm")
sns.catplot(x='Algorithm', y='value', hue='variable', data=performance_df, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
#plt.show()

#we are trying to improve the model
#1. Change the max_features parameter in tfidf we change it to 3000 : tfidf = TfidfVectorizer(max_features=3000) performance increased.
#2. We tried Scaling but not used because performance is not increased
#3. We tried vectorising but not used because performance is not increased
#4. We tried to use voting classified(combination of best performing models) but performance not improved


#voting classifier 1
svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

voting = VotingClassifier(estimators=[('svc', svc), ('mnb', mnb), ('etc', etc)], voting='soft')

voting.fit(X_train, y_train)

VotingClassifier(estimators=[('svc', SVC(kernel='sigmoid', gamma=1.0, probability=True)),('mnb', MultinomialNB()), ('etc', ExtraTreesClassifier(n_estimators=50, random_state=2))], voting='soft')

y_pred = voting.predict(X_test)
print("Accuracy1", accuracy_score(y_test,y_pred))
print("Precision1", precision_score(y_test,y_pred))



#voting classifier 2
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
mnb = MultinomialNB()
knc = KNeighborsClassifier()


voting = VotingClassifier(estimators=[('rfc', rfc), ('mnb', mnb), ('knc', etc)], voting='soft')

voting.fit(X_train, y_train)

VotingClassifier(estimators=[('rfc', RandomForestClassifier(n_estimators=50, random_state=2)),('mnb', MultinomialNB()), ('knc', KNeighborsClassifier())], voting='soft')

y_pred = voting.predict(X_test)
print("Accuracy2", accuracy_score(y_test,y_pred))
print("Precision2", precision_score(y_test,y_pred))


#voting classifier 3
mnb = MultinomialNB()
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
svc = SVC(kernel='sigmoid', gamma=1.0, probability=True)


voting = VotingClassifier(estimators=[('mnb', mnb), ('rfc', rfc), ('svc', svc)], voting='soft')

voting.fit(X_train, y_train)

VotingClassifier(estimators=[('mnb', MultinomialNB()),('rfc', RandomForestClassifier(n_estimators=50, random_state=2)), ('svc', SVC(kernel='sigmoid', gamma=1.0, probability=True))], voting='soft')

y_pred = voting.predict(X_test)
print("Accuracy3", accuracy_score(y_test,y_pred))
print("Precision3", precision_score(y_test,y_pred))

#5. We tried to apply stacking but performance is not increased

estimators = [('svc', svc), ('mnb', mnb), ('etc', etc)]
final_estimator = RandomForestClassifier()
clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy4", accuracy_score(y_test, y_pred))
print("Precision4", precision_score(y_test, y_pred))
'''

#we are choosing tfidf --> mnb #our best performing model

#creating pipeline
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(mnb, open('model.pkl', 'wb'))
