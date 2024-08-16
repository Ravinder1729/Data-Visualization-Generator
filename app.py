from flask import Flask, request, render_template, jsonify
import openai
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import io
import os

app = Flask(__name__)

# Load API keys
with open('key/key.txt') as f:
    openai.api_key = f.read().strip()

google_api_key = 'AIzaSyB1ENyockmC2sWDs0XprPpBeFCdTwWY6dw'  

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    # Read the CSV file
    df = pd.read_csv(file)
    if df.empty:
        return jsonify({'error': 'The uploaded CSV file is empty. Please upload a valid CSV file with data.'}), 400

    # Save the dataframe as a CSV to use with the agent
    file_path = os.path.join('uploads', file.filename)
    df.to_csv(file_path, index=False)

    return jsonify({'message': 'File uploaded successfully', 'file_path': file_path}), 200

@app.route('/submit', methods=['POST'])
def submit_query():
    user_input = request.form.get('query')
    file_path = request.form.get('file_path')

    if not user_input or not file_path:
        return jsonify({'error': 'Query and file path are required'}), 400

    # Create the CSV agent using the uploaded file
    agent = create_csv_agent(
        ChatGoogleGenerativeAI(google_api_key=google_api_key, model='gemini-1.5-pro-latest'),
        file_path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        allow_dangerous_code=True
    )

    result = agent.invoke(user_input)
    
    # Generate visualization code using GPT-3.5-turbo
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """You are a helpful AI assistant. The data has been loaded into a DataFrame named 'df'. 
            The columns are: 'Invoice ID', 'Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Unit price', 'Quantity', 'Tax 5%', 
            'Total', 'Date', 'Time', 'Payment', 'cogs', 'gross margin percentage', 'gross income', 'Rating'. 
            Please generate Python visualization code using plotly with different colors and the DataFrame 'df' without any syntax errors. 
            Don't generate any docstring or triple quotes in the code. try to add some different colors in my visualization"""},
            {"role": "user", "content": user_input}
        ]
    )

    if response.choices:
        plotly_code = response.choices[0].message['content']

        # Prepare a context for the execution of the generated code
        exec_context = {
            "df": pd.read_csv(file_path),
            "pd": pd,
            "px": px,
            "go": go
        }

        # Execute the generated Plotly code
        try:
            exec(plotly_code, exec_context)
            plot_html = exec_context['fig'].to_html(full_html=False)
            return jsonify({'result': result['output'], 'plot_html': plot_html}), 200
        except Exception as e:
            return jsonify({'error': f"Error in executing Plotly code: {e}"}), 500

    return jsonify({'error': 'Failed to generate code'}), 500

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
