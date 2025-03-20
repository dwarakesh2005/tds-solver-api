from flask import Flask, request, jsonify
import os
import zipfile
import pandas as pd
import tempfile
import json
from openai import OpenAI
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure OpenAI with AI Proxy using the existing OPENAI_API_KEY
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),  # Using existing environment variable
    base_url="https://aiproxy.sanand.workers.dev/openai/v1"  # Added /v1 to the base URL
)



@app.route('/api', methods=['POST'])
def process_question():
    try:
        # Get the question from the request
        question = request.form.get('question')
        
        # Check if a file was uploaded
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Create a temporary directory to store files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save the uploaded file
                    file_path = os.path.join(temp_dir, secure_filename(file.filename))
                    file.save(file_path)
                    
                    # If it's a zip file, extract it
                    if file.filename.endswith('.zip'):
                        extract_dir = os.path.join(temp_dir, 'extracted')
                        os.makedirs(extract_dir, exist_ok=True)
                        
                        with zipfile.ZipFile(file_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                        
                        # Look for CSV files in the extracted directory
                        for root, _, files in os.walk(extract_dir):
                            for filename in files:
                                if filename.endswith('.csv'):
                                    csv_path = os.path.join(root, filename)
                                    df = pd.read_csv(csv_path)
                                    
                                    # Check if the question is about finding a value in the "answer" column
                                    if "answer column" in question.lower():
                                        if "answer" in df.columns:
                                            answer = df["answer"].iloc[0]
                                            return jsonify({"answer": str(answer)})
                                    
                                    # Otherwise, use AI Proxy to analyze the CSV data
                                    # Note: AI Proxy only supports gpt-4o-mini model
                                    csv_data = df.head(10).to_json(orient="records")
                                    response = client.chat.completions.create(
                                        model="gpt-4o-mini",
                                        messages=[
                                            {"role": "system", "content": "You are a helpful assistant that answers questions about data files."},
                                            {"role": "user", "content": f"I have a CSV file with the following data (showing first 10 rows): {csv_data}. The file has {len(df)} rows and {len(df.columns)} columns. {question}"}
                                        ]
                                    )
                                    answer = response.choices[0].message.content.strip()
                                    return jsonify({"answer": answer})
        
        # If no file or no CSV found, just use AI Proxy to answer the question
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for a Tools in Data Science course. Answer the question accurately and concisely."},
                {"role": "user", "content": question}
            ]
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"answer": answer})
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
