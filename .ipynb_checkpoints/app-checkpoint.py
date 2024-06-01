from flask import Flask, request, jsonify
import pickle
import category_encoders as ce
import numpy as np
import pandas as pd
from email.utils import parseaddr

app = Flask(__name__)

# Load your trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the TLD_Freq data from the CSV file
tld_data = pd.read_csv('tld_data_selected.csv')

# Load the trained binary encoder (saved during training)
with open('binary_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Define your preprocessing function
def preprocess_data(data):
    # Extract features from the data dictionary
    attachment_count = data['Attachment Count']
    attachment_extensions = data['Attachment Extension']  # Keep it as a raw string
    email_subject = data['Email Subject']

        # Convert 'Attachment Count' to integer (handle non-numeric values)
    try:
        attachment_count = int(attachment_count)
    except ValueError:
        attachment_count = 0  # Set to 0 if not a valid integer

    # Extract TLD from email addresses
    def extract_tld(email):
        email = email.replace('"', '')  # Remove double quotes
        _, address = parseaddr(email)
        domain = address.split('@')[-1] if '@' in address else ''
        tld = domain.split('.')[-1] if domain else ''
        return tld

    email_from = data.get('Email From', '')  # Handle missing 'Email From' gracefully
    tld = extract_tld(email_from)

    # Get the TLD_Freq corresponding to the extracted TLD
    tld_freq = tld_data[tld_data['TLD'] == tld]['TLD_Freq'].values[0]

    # Calculate the length of the email address
    email_from_length = len(email_from)
    
# Explode the 'Attachment Extension' values
    df_extension = pd.DataFrame({"Attachment Extension": attachment_extensions.split(",")})
    df_dict = df_extension.to_dict(orient='records')

# Return the preprocessed features as a dictionary
    return {
    'Attachment Count': attachment_count,
    'Attachment Extensions': df_dict,
    'Email Subject': email_subject,
    'TLD_Freq': tld_freq,
    'Email_From_Length': email_from_length
    }
    #df_extension = df_extension.explode("Attachment Extension")
     # Apply binary encoding to the exploded extensions
    #df_encoded_extension = encoder.transform(df_extension)
     # Print the encoded DataFrame for inspection
    # Return the preprocessed features
    return {
        'Attachment Count': attachment_count,
        'Attachment Extensions': df_extension,
        'Email Subject': email_subject,
        'TLD_Freq': tld_freq,
        'Email_From_Length': email_from_length
    }

@app.route('/preprocess', methods=['POST'])  # Changed route to /preprocess
def preprocess():
    try:
        # Get data from POST request
        data = request.get_json()

        # Preprocess the data
        preprocessed_features = preprocess_data(data['features'])

        # Return preprocessed features (no prediction)
        return jsonify({'preprocessed_features': preprocessed_features})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
