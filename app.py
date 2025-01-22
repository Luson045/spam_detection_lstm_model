import streamlit as st
import tensorflow as tf
import re
import textwrap
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot

# Load the TensorFlow model
model = tf.keras.models.load_model('models/my_model.h5')  # Make sure your model is in the same directory or provide the correct path

vocab = 5000
length = 30

# Function to split text into sentences based on punctuation
def split_to_sentences(text, max_length=30):
    split_text = re.split(r'[,.?!]', text)
    final_sentences = []
    for sentence in split_text:
        sentence = sentence.strip()
        if sentence:
            if len(sentence) > max_length:
                wrapped_sentences = textwrap.wrap(sentence, width=max_length, break_long_words=False, break_on_hyphens=False)
                final_sentences.extend(wrapped_sentences)
            else:
                final_sentences.append(sentence)
    return final_sentences

# Function to process and predict based on the input string
def tell(string):
    ns = one_hot(string, vocab)  # One-hot encoding for the input string
    padded_s = pad_sequences([ns], padding='pre', maxlen=length)
    pred = model.predict(padded_s)
    return pred[0][0]

def predict(string):
    result = split_to_sentences(string, max_length=30)
    answer = []
    for i in result:
        t = tell(i)
        answer.append(t)
    if max(answer) >= 0.3:
        return "Spam", result[answer.index(max(answer))]
    else:
        return "Not Spam", None

# Streamlit app layout
st.title("Text Spam Detection")
st.write("Enter a news or text to analyze whether it is spam or not:")

# Text input from the user
input_text = st.text_area("Enter Text", "Type your text here...")

if st.button("Analyze"):
    if input_text:
        st.write("Analyzing the input text...")
        prediction, spam_word = predict(input_text)
        st.write(f"Prediction: {prediction}")
        if spam_word:
            st.write(f"Spam word: {spam_word}")
    else:
        st.warning("Please enter some text!")

