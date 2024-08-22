Workflow of the Task

Data Download:

First, I download the data from the given link.
The data is stored in a folder, and within that folder, there are subfolders for each category.

Fetch Resume Files:

Write a code snippet to fetch all the resume PDF files from each folder, along with the corresponding category.
Store them in a Pandas DataFrame.
I saved the DataFrame as cleaned_data.csv.

Data Preprocessing:

With a user-defined function, I preprocess the resume text.

Model Development:

I developed a couple of machine learning models for resume classification.
As developing an NLP text classification model from scratch is a very cumbersome process, I fine-tuned an existing BERT model, specifically DistilBERT, for text classification.
My whole fine-tuning notebook is available here.
I stored the saved model in my Google Drive: Download Link.

Resume Classification:

I used the fine-tuned model for resume classification with higher accuracy.

Script Development:

I wrote a script named script.py.
In this script, I first gather all the data in a single folder named full_data, where I store all the resume files.
The script takes each file from full_data, preprocesses the text inside that file, and makes predictions.
Based on the predicted category, it creates a folder and stores the resume file in that folder.
After categorizing all the files, it creates a CSV file named categorized_resume as suggested by the task.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

