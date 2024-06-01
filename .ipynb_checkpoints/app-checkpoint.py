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
    spam = 'Spam'
   # df_api = {'Attachment Count': attachment_count,
    #    'Email Subject': email_subject,
     #   'TLD_Freq': tld_freq,
      #  'Email_From_Length':email_from_length}
    df_api = pd.DataFrame({
        'Attachment Count': [attachment_count],
        'Email Subject': [email_subject],
        'closeNotes':[spam],
        'Email_From_Length': [email_from_length],
        'TLD_Freq': [tld_freq]
    })

    if attachment_count > 0 and attachment_extensions != "":
        df_new = pd.DataFrame(np.repeat(df_api.values, attachment_count, axis=0))
        df_new.columns = df_api.columns

    elif attachment_count > 0 and attachment_extensions == "":
        df_new = df_api
    else:
        df_new = df_api
    


    df_extension = pd.DataFrame({"Attachment Extension": attachment_extensions.split(",")})
    # Create a DataFrame with a single row for the 'Attachment Extension'
    df_extension = pd.DataFrame({"Attachment Extension": attachment_extensions.split(",")})
    merged_df = pd.concat([df_new, df_extension], axis=1)
    # Display the merged DataFrame
    #print(merged_df)
    # Apply binary encoding to the 'Attachment Extension'
    df_encoded = encoder.transform(merged_df)
    df_encoded = df_encoded[['Attachment Count','Attachment Extension_0','Attachment Extension_1','Attachment Extension_2','Attachment Extension_3','Attachment Extension_4',
                             'Attachment Extension_5','Attachment Extension_6','Attachment Extension_7','Email Subject','Email_From_Length','TLD_Freq']]



    print(df_encoded)

#Natural Language Preprocessing for Email Subject







# Return the preprocessed features as a dictionary
    return {
    'Attachment Count': attachment_count,
    'Email Subject': email_subject, # Assuming 'spam' is a predefined value
    'Email_From_Length': email_from_length,
    'TLD_Freq': tld_freq
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
