import streamlit as st
from PIL import Image
import pickle
import string
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
import time

if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.error("Please log in to access this page.")
    st.stop()

ps = PorterStemmer()
image = Image.open('icons/logo.png')

st.set_page_config(page_title="Cyberbullying Detection", page_icon=image)

# Hide Streamlit menu and footer
hide_menu = """
<style>
#MainMenu {
    visibility: hidden;
}
footer {
    visibility: hidden;
}
</style>
"""
st.markdown(hide_menu, unsafe_allow_html=True)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image(image, use_column_width=True, output_format='auto')
st.sidebar.markdown("---")
st.sidebar.markdown("<br> <br> <br> <br> <br> <br> <h1 style='text-align: center; font-size: 18px; color: #0080FF;'>© 2023 | Ioannis Bakomichalis</h1>", unsafe_allow_html=True)

def clean_text(tweet):
    tweet = re.sub(r'http\S+', '', tweet)  # remove URL
    tweet = re.sub(r'@[^\s]+[\s]?', '', tweet)  # Remove usernames
    tweet = re.sub(r'#[^\s]+[\s]?', '', tweet)  # Remove hashtags
    tweet = re.sub(r':[^\s]+[\s]?', '', tweet)  # Remove emojis
    tweet = re.sub('[^ a-zA-Z0-9]', '', tweet)  # remove special characters
    tweet = re.sub('RT', '', tweet)  # remove RT
    tweet = re.sub('[0-9]', '', tweet)  # remove Numbers
    return tweet

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

tfidf = pickle.load(open('pickle/TFIDFvectorizer.pkl', 'rb'))
model = pickle.load(open('pickle/bestmodel.pkl', 'rb'))

st.title("Cyber-Bullying Detection🔍")
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)
input_text = st.text_area("Enter the text to analyze", key="Enter the text to analyze")
col1, col2 = st.columns([1, 6])
with col1:
    button_predict = st.button('Predict')
with col2:
    def clear_text():
        st.session_state["Enter the text to analyze"] = ""
    button_clear = st.button("Clear", on_click=clear_text)

st.markdown("---")

if button_predict:
    if input_text == "":
        st.snow()
        st.warning("Please provide some text!")
    else:
        with st.spinner("Prediction** in progress. Please wait 🙏"):
            time.sleep(3)

        cleanText = clean_text(input_text)
        transformText = transform_text(cleanText)

        vector_input = tfidf.transform([transformText])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.subheader("Result")
            st.error(":red[Cyberbullying]")
        else:
            st.subheader("Result")
            st.success(":green[Not Cyberbullying]")
        st.markdown("---")
        st.subheader("Original Text")
        expander_original = st.expander("Information", expanded=False)
        with expander_original:
            st.info("The text that the user provided!")
        st.text(input_text)
        st.markdown("---")
        st.subheader("Cleaned Text")
        expander_clean = st.expander("Information", expanded=False)
        with expander_clean:
            st.info("From original text has removed punctuation and special characters. Also it has removed hashtags, tags and emojis!")
        st.text(cleanText)
        st.markdown("---")
        st.subheader("Transformed Text")
        expander_transform = st.expander("Information", expanded=False)
        with expander_transform:
            st.info("From Cleaned text has removed stopwords and transformed to lowercase. Also, it has used Stemming!")
        st.text(transformText)
        st.markdown("---")
        st.subheader("Binary Prediction")
        expander_binary = st.expander("Information", expanded=False)
        with expander_binary:
            st.info("Binary Prediction from the Model!")
        if result == 1:
            st.markdown(":red[" + str(result) + "]")
        else:
            st.markdown(":green[" + str(result) + "]")
        st.markdown("---")
        st.subheader("Model Accuracy")
        expander_accuracy = st.expander("Information", expanded=False)
        with expander_accuracy:
            st.info("Model Accuracy using Random Forest (RF) Classifier!")
        st.warning("Accuracy:  *91.70 %*")
        st.markdown("---")