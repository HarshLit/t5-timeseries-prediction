import pandas as pd
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained("t5-small")

# 1. Load the data using pandas
data = pd.read_csv('banknifty 1 min.csv') # Replace 'your_file.csv' with the name of your CSV file

# 2. Clean the data (optional)
# Perform any data cleaning or filtering here, if needed

# 3. Tokenize the data
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# 4. Create input-output pairs
def convert_data_to_text_format(row):
    # Combine the row values into a single string, e.g., "time: 2000-01-03 09:15:00+05:30, open: 1054.8101, ..."
    input_text = ', '.join([f"{col}: {row[col]}" for col in data.columns if col != 'open'])
    
    output_text = str(row['open'])
    
    # Use the tokenizer.encode_plus method to convert the input and output text into tokenized format
    tokenized_input = tokenizer.encode_plus(input_text, max_length=512, return_tensors='pt', padding='max_length', truncation=True)
    tokenized_output = tokenizer.encode_plus(output_text, max_length=32, return_tensors='pt', padding='max_length', truncation=True)
    
    return {
        'input_ids': tokenized_input['input_ids'].squeeze(),
        'attention_mask': tokenized_input['attention_mask'].squeeze(),
        'labels': tokenized_output['input_ids'].squeeze()
    }

# Apply the function to the DataFrame
tokenized_data = data.apply(convert_data_to_text_format, axis=1)
tokenized_data.to_pickle('tokenized_data.pkl')

# 5. Save the preprocessed data
# Save the tokenized data as a pickle file or another format of your choice