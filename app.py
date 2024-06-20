from fastapi import FastAPI
from pydantic import BaseModel
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Initialize the FastAPI app
app = FastAPI()

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained(r"D:\All_data_science_project\NLP\fine-tune-2\saved_models")
model = DistilBertForSequenceClassification.from_pretrained(r"D:\All_data_science_project\NLP\fine-tune-2\saved_models")

# Set the device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Label mapping
label_map = {
    0: 'business',
    1: 'entertainment',
    2: 'politics',
    3: 'sport',
    4: 'tech'
}

# Define the request body
class TextInput(BaseModel):
    text: str

# Define a root endpoint for health check
@app.get("/")
async def root():
    return {"message": "Welcome to the Text Classification API"}

@app.post("/predict")
async def predict(text_input: TextInput):
    user_input = text_input.text
    # Tokenize the input text
    predict_input = tokenizer.encode(
        user_input,
        truncation=True,
        padding=True,
        return_tensors='pt'  # Use PyTorch tensors
    )

    # Move the input tensor to the same device as the model
    predict_input = predict_input.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(predict_input)

    # Get the predicted label
    prediction_value = torch.argmax(output.logits, axis=1).item()

    # Map the predicted label to the corresponding category
    predicted_category = label_map[prediction_value]

    return {"predicted_category": predicted_category}
