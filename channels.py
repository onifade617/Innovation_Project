# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 08:51:04 2024

@author: SAIL
"""

import streamlit as st
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score


st.title("Application of Natural Language Processing Models on Political Information")
#import our data
penguins_df = pd.read_csv("CHANNELS_DATA.csv")

penguins_df = penguins_df.drop(penguins_df.columns[0], axis=1)
st.dataframe(penguins_df)
selected_title = st.selectbox('Select a Title:', penguins_df['Title'].unique())

# Get the content corresponding to the selected title
selected_content = penguins_df.loc[penguins_df['Title'] == selected_title, 'Content'].iloc[0]


# Display the content within a larger textarea
st.text_area('Content:', selected_content, height=600)

# Button to perform sentiment analysis
if st.button('Perform Sentiment Analysis'):
    # Sentiment analysis
    model = pipeline("sentiment-analysis")
    # Truncate the input sequence if it exceeds the maximum supported length
    max_seq_length = model.tokenizer.model_max_length
    truncated_content = selected_content[:max_seq_length]
    result = model(truncated_content)
    
    # Display sentiment analysis results
    st.write("Sentiment:", result[0]["label"])
    st.write("Confidence:", result[0]["score"])

# Button to perform named entity recognition
if st.button('Perform Named Entity Recognition'):
    # Named entity recognition
    ner_tagger = pipeline("ner", aggregation_strategy="simple")
    outputs = ner_tagger(selected_content)
    
    # Convert predictions to DataFrame
    df_entities = pd.DataFrame(outputs)
    st.write(df_entities)
    
# Button to perform text summarization
if st.button('Perform Text Summarization'):
    # Text summarization
    summarizer = pipeline("summarization")
    outputs = summarizer(selected_content, max_length=56, clean_up_tokenization_spaces=True)
    
    # Display text summarization output
    st.write("Summary:")
    st.write(outputs)
    
    
# Get user input for question
question = st.text_input('Enter your question:', 'What does the user wants to know about the selected topic?')

# Button to perform question answering
if st.button('Submit'):
    # Perform question answering
    reader = pipeline("question-answering",model="dmis-lab/biobert-large-cased-v1.1-squad")
    output = reader(question=question, context=selected_content)
    
    # Display the answer
    st.write("Answer:")
    st.write(output['answer'])
    
    
