import os
import re
import sys
import shutil
import pandas as pd
import fitz  
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import torch


"""
First i gather all my resume pdf file in a single folder  name d "full data". then i run this script on this folder . 
this scrript is taking each resume file frm this folder and make prediction on that particular resume and as the predicted category 
is the category of the resume . it is saving the resume file in a folder named as the predicted category . 
example : like a resume file name is 10876132.pdf . this script will make prediction on this and the predicted category is AUTOMOBILE
then it will create a folder named AUTOMOBILE and save this file in this foldder ,
"""


# Load the tokenizer and model
tokenizer_fine_tuned = DistilBertTokenizer.from_pretrained(r"D:\All_data_science_project\NLP\fine-tune-2\My_New_Saved_Model")
model_fine_tuned = DistilBertForSequenceClassification.from_pretrained(r"D:\All_data_science_project\NLP\fine-tune-2\My_New_Saved_Model")


label_map = {
    0: 'ACCOUNTANT',
    1: 'ADVOCATE',
    2: 'AGRICULTURE',
    3: 'APPAREL',
    4: 'ARTS',
    5: 'AUTOMOBILE',
    6: 'AVIATION',
    7: 'BANKING',
    8: 'BPO',
    9: 'BUSINESS-DEVELOPMENT',
    10: 'CHEF',
    11: 'CONSTRUCTION',
    12: 'CONSULTANT',
    13: 'DESIGNER',
    14: 'DIGITAL-MEDIA',
    15: 'ENGINEERING',
    16: 'FINANCE',
    17: 'FITNESS',
    18: 'HEALTHCARE',
    19: 'HR',
    20: 'INFORMATION-TECHNOLOGY',
    21: 'PUBLIC-RELATIONS',
    22: 'SALES',
    23: 'TEACHER'
}


def preprocess(sentence):
    sentence = str(sentence).lower()
    sentence = sentence.replace('{html}',"")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '', cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]

    return " ".join(filtered_words)


def predict_category(text):
    inputs = tokenizer_fine_tuned(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_fine_tuned(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    prediction_value = predictions.item()
    predicted_category = label_map[prediction_value]
    return predicted_category




def extract_text_from_pdf(pdf_path):
    extracted_text = ""
    try:
        with fitz.open(pdf_path) as pdf_document:
            for page in pdf_document:
                extracted_text += page.get_text()
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
    return extracted_text


def categorize_resumes(directory):
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        sys.exit(1)

    print(f"Processing files in directory: {directory}")

   
    categorized_data = []

    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            print(f"Processing file: {file_path}")
            
            if filename.endswith(".pdf"):  
                
                extracted_text = extract_text_from_pdf(file_path)
                if not extracted_text:
                    print(f"No text are extracted from {filename}. Skipping...")
                    continue
                
                
                preprocessed_content = preprocess(extracted_text)
                
                
                category = predict_category(preprocessed_content)
                print(f"Predicted category for {filename}: {category}")

                
                category_folder = os.path.join(directory, str(category))
                if not os.path.exists(category_folder):
                    os.makedirs(category_folder)

               
                shutil.move(file_path, os.path.join(category_folder, filename))

                
                categorized_data.append([filename, category])
            else:
                print(f"Skip non-PDF file: {filename}")

    
    df = pd.DataFrame(categorized_data, columns=["File Name", "Category"])
    csv_file_path = os.path.join(directory, "categorize_resumes.csv")
    df.to_csv(csv_file_path, index=False)

    print(f"Categorization is completed. Results are saved to {csv_file_path} !!!!!!!!!")
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    
    input_directory = sys.argv[1]
    categorize_resumes(input_directory)
