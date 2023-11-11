from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pickle  
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

model_path = '/home/phani/Desktop/bert/weights_emotion/' 
model = BertForSequenceClassification.from_pretrained(model_path)
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)

label_encoder_path = '/home/phani/Desktop/bert/weights_emotion/label_encoder.pkl' 
with open(label_encoder_path, 'rb') as le_file:
    label_encoder = pickle.load(le_file)

# Function to perform inference
def predict_emotion(text):
    tokenized_text = tokenizer.encode_plus(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    with torch.no_grad():
        logits = model(**tokenized_text).logits
        _, predicted_class = torch.max(logits, dim=1)

    # Convert the predicted class index back to the original label
    predicted_label = label_encoder.inverse_transform(predicted_class.cpu().numpy())[0]
    return predicted_label

# Define an endpoint for inference
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']
    result = predict_emotion(text)
    return jsonify({'emotion': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
