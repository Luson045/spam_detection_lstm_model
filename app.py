import streamlit as st
import tensorflow as tf
import numpy as np
import re
import textwrap
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Constants
VOCAB_SIZE = 5000
MAX_LENGTH = 30

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('my_model.h5')

def split_to_sentences(text, max_length=30):
    # Split the text on specified punctuation marks
    split_text = re.split(r'[,.?!]', text)
    
    final_sentences = []
    for sentence in split_text:
        sentence = sentence.strip()
        if sentence:
            if len(sentence) > max_length:
                wrapped_sentences = textwrap.wrap(sentence, 
                                                width=max_length, 
                                                break_long_words=False, 
                                                break_on_hyphens=False)
                final_sentences.extend(wrapped_sentences)
            else:
                final_sentences.append(sentence)
    return final_sentences

def tell(string, model):
    # Convert string to integer sequence using one_hot
    ns = tf.keras.preprocessing.text.one_hot(string, VOCAB_SIZE)
    # Pad sequence
    padded_s = tf.keras.preprocessing.sequence.pad_sequences([ns], padding='pre', maxlen=MAX_LENGTH)
    # Predict
    pred = model.predict(padded_s, verbose=0)
    return pred[0][0]

def analyze_text(text, model):
    sentences = split_to_sentences(text, max_length=MAX_LENGTH)
    scores = []
    
    for sentence in sentences:
        score = tell(sentence, model)
        scores.append(float(score))
    
    return sentences, scores

def main():
    st.title("Spam Text Analyzer")
    
    # Load model
    try:
        model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Text input
    text_input = st.text_area("Enter the text to analyze:", height=150)
    
    if st.button("Analyze"):
        if text_input:
            # Analysis
            sentences, scores = analyze_text(text_input, model)
            
            # Overall spam probability
            max_score = max(scores)
            is_spam = max_score >= 0.3
            
            # Results section
            st.header("Analysis Results")
            
            # Spam verdict
            st.subheader("Verdict")
            if is_spam:
                st.error("⚠️ This text is likely SPAM!")
            else:
                st.success("✅ This text appears to be legitimate.")
            
            # Statistics
            st.subheader("Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Maximum Spam Score", f"{max_score:.2%}")
            with col2:
                st.metric("Average Spam Score", f"{np.mean(scores):.2%}")
            with col3:
                st.metric("Number of Segments", len(sentences))
            
            # Detailed analysis
            st.subheader("Segment Analysis")
            
            # Create DataFrame for visualization
            df = pd.DataFrame({
                'Segment': sentences,
                'Spam Score': scores
            })
            
            # Bar chart
            fig_bar = px.bar(
                df,
                y='Spam Score',
                title='Spam Scores by Text Segment',
                labels={'Spam Score': 'Probability of Spam'},
                height=400
            )
            fig_bar.add_hline(y=0.3, line_dash="dash", line_color="red", 
                            annotation_text="Spam Threshold (0.3)")
            st.plotly_chart(fig_bar)
            
            # Line chart
            fig_line = px.line(
                df,
                y='Spam Score',
                title='Spam Score Trend',
                labels={'index': 'Segment Number', 'Spam Score': 'Probability of Spam'},
                height=400
            )
            fig_line.add_hline(y=0.3, line_dash="dash", line_color="red",
                             annotation_text="Spam Threshold (0.3)")
            st.plotly_chart(fig_line)
            
            # Detailed segments table
            st.subheader("Detailed Segment Analysis")
            df['Spam Score'] = df['Spam Score'].apply(lambda x: f"{x:.2%}")
            st.dataframe(df)
            
            # Most suspicious segment
            if is_spam:
                max_score_idx = scores.index(max_score)
                st.warning(f"Most suspicious segment: \"{sentences[max_score_idx]}\" (Score: {max_score:.2%})")
        else:
            st.warning("Please enter some text to analyze.")

if __name__ == "__main__":
    main()