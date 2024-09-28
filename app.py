import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

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

# Inject custom CSS
st.markdown(
    """
    <style>
    /* Apply dark theme to entire app */
    .css-1y0tad9 { 
        background-color: #121212; /* Dark background color */ 
        color: #e0e0e0; /* Light text color */ 
    }

    /* Style buttons */
    .stButton > button {
        background-color: #333; 
        color: #e0e0e0; 
        border: none; 
    }

    .stButton > button:hover {
        background-color: #444; 
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("Email/SMS Spam Classifier📨")
input_sms = st.text_input("Enter the Message")

if st.button('Predict'):

    #1. Preproccesing
    transformed_sms = transform_text(input_sms)

    #2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    #3. Predict
    result = model.predict(vector_input)[0]

    #4. Diploy
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
