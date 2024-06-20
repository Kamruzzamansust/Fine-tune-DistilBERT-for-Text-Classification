from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the tokenizer and model
tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(r"D:\All_data_science_project\NLP\fine-tune-2\saved_models")
model_fine_tuned = DistilBertForSequenceClassification.from_pretrained(r"D:\All_data_science_project\NLP\fine-tune-2\saved_models")

# Set the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_fine_tuned.to(device)

# Label mapping
label_map = {
    0: 'business',
    1: 'entertainment',
    2: 'politics',
    3: 'sport',
    4: 'tech'
}

# Streamlit app
st.title('Text Classification with DistilBERT')

# User input
user_input = st.text_area("Enter text to classify:", height=200)

# Predict button
if st.button("Predict"):
    if user_input.strip() != "":
        # Tokenize the input text
        predict_input = tokenizer_fine_tuned.encode(
            user_input,
            truncation=True,
            padding=True,
            return_tensors='pt'  # Use PyTorch tensors
        )
        
        # Move the input tensor to the same device as the model
        predict_input = predict_input.to(device)
        
        # Make the prediction
        with torch.no_grad():
            output = model_fine_tuned(predict_input)
        
        # Get the predicted label
        prediction_value = torch.argmax(output.logits, axis=1).item()
        
        # Map the predicted label to the corresponding category
        predicted_category = label_map[prediction_value]
        
        # Display the predicted category
        st.write(f"Predicted category: {predicted_category}")
    else:
        st.write("Please enter some text to classify.")
