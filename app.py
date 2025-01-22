import streamlit as st
import tensorflow as tf
import re
import textwrap
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import matplotlib.pyplot as plt

# Load the TensorFlow model
model = tf.keras.models.load_model('models/my_model.h5')  # Make sure your model is in the correct directory

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
    return result, answer

# Streamlit app layout
st.set_page_config(page_title="Spam Detection", page_icon="🔍", layout="wide")
st.title("📬 Text Spam Detection")
st.write("Enter the news or text below to analyze whether it is spam or not:")

# Create an input section with a stylish text area
input_text = st.text_area("Type the text here", "Type your text to check if it's spam...", height=200)

# Create a button to trigger the analysis
analyze_button = st.button("Analyze")

if analyze_button:
    if input_text:
        # Show a loading spinner while analyzing
        with st.spinner("Analyzing... Please wait."):
            # Get sentences and their respective spam probabilities
            sentences, probabilities = predict(input_text)

            # Determine if it's spam or not
            max_prob = max(probabilities)
            prediction = "Spam" if max_prob >= 0.3 else "Not Spam"

            # Display the main prediction (spam or not)
            st.subheader(f"**Prediction:** {prediction}")
            st.write(f"**Maximum Spam Probability:** {max_prob * 100:.2f}%")

            # Display sentence with the highest probability of being spam
            if prediction == "Spam":
                spam_sentence = sentences[probabilities.index(max_prob)]
                st.write(f"**Spam Sentence:** {spam_sentence}")
                st.write(f"**Spam Probability:** {max_prob * 100:.2f}%")

            # Display a table of sentences with probabilities
            st.write("### Sentence Probabilities:")
            sentence_data = {
                "Sentence": sentences,
                "Spam Probability (%)": [prob * 100 for prob in probabilities]
            }
            st.dataframe(sentence_data)

            # Display a bar chart with probabilities of each sentence
            st.write("### Spam Probability Bar Chart:")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(sentences, [prob * 100 for prob in probabilities], color='tomato')
            ax.set_xlabel("Spam Probability (%)")
            ax.set_title("Spam Probability of Each Sentence")
            ax.set_xlim(0, 100)
            st.pyplot(fig)

    else:
        st.warning("Please enter some text!")
