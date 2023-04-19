import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Step 1: Load the trained model
checkpoint_path = "C:/Users/91904/Documents/Codes/Training AI/lightning_logs/version_0/checkpoints/epoch=2-step=2091.ckpt"
tokenizer = T5Tokenizer.from_pretrained('t5-small')
checkpoint = torch.load(checkpoint_path)
model = T5ForConditionalGeneration.from_pretrained('t5-small', state_dict=checkpoint['state_dict'])

# Assuming test_data is a DataFrame with 'time', 'open', 'high', 'low', 'close', and 'weekday' columns
test_data = pd.read_pickle("tokenized_data_test.pkl")

# Step 2: Generate predictions
def predict(model, tokenizer, input_text):
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    output_tokens = model.generate(input_tokens)
    output_text = tokenizer.decode(output_tokens[0])
    return output_text

predictions = []
for index, row in test_data.iterrows:
    input_text = f"time {row['time']} open {row['open']} high {row['high']} low {row['low']} close {row['close']}"
    prediction = predict(model, tokenizer, input_text)
    predictions.append(float(prediction.split(' ')[-1]))

# Step 3: Evaluate the model
# Assuming the ground_truth contains the true future values of the 'open' column
ground_truth = test_data['open']

accuracy = accuracy_score(ground_truth, predictions)
precision = precision_score(ground_truth, predictions, average='weighted')
recall = recall_score(ground_truth, predictions, average='weighted')
f1 = f1_score(ground_truth, predictions, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")