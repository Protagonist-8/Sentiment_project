import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
import re
import joblib

# Load model + vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
nltk.download('stopwords')
port_stem=PorterStemmer()
# Preprocessing function (same as training!)
def stemming(content):
  stemmed_content=re.sub('[^a-zA-Z]',' ',content)
  stemmed_content=stemmed_content.lower()
  stemmed_content=stemmed_content.split()
  stemmed_content=[port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
  stemmed_content=' '.join(stemmed_content)

  return stemmed_content

# Streamlit UI
st.title("Sentiment Analysis App")

user_input = st.text_area("Enter text here:")

if st.button("Predict"):
    if user_input.strip():
        # Preprocess
        cleaned_text = stemming(user_input)

        # Feature extraction
        features = tfidf.transform([cleaned_text])

        # Prediction
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

        if(pred == 1):
            pred = "Positive"
            st.write(f"Wow, someone is in a good mood Today!!!")
            st.write(f"**I can say this with:** {max(prob):.2f} confidence")
        else:
            pred = "Negative"
            st.write(f"Uh Oh, someone is in a bad mood Today!!!, Don't worry, things will get better.")
            st.write(f"**I can say this with:** {max(prob):.2f} confidence")
    else:
        st.write("Please enter some text.")
