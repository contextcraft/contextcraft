# ChatGPT-4 Method Name Prediction

This project predicts method names based on input prompts using OpenAI's **ChatGPT-4** model. It reads prompts from a CSV file, generates method names or other relevant predictions, and saves the outputs to a new CSV file.

## Project Structure

The project is organized into several modules, with a clear separation of concerns:

```bash
chatgpt4_method_prediction
main.py # Main script to run the process 
chatgpt4_config.py # Contains ChatGPT4Configuration class 
text_processing.py # Contains TextProcessing class 
csv_handler.py # Handles reading and writing CSV files 
method_name_pred.py # Orchestrates the process using the above classes
```

### Project Modules

- **ChatGPT4Configuration**: Handles setting up the OpenAI API client and sending prompts to the ChatGPT-4 model.
- **TextProcessing**: Manages input prompts and retrieves responses from ChatGPT-4.
- **CSVHandler**: Reads prompts from a CSV file and writes generated outputs back to a CSV.
- **MethodNamePrediction**: Coordinates the overall process, connecting the ChatGPT-4 API, CSV handling, and text processing.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- A valid OpenAI API key with access to ChatGPT-4

### Step 1: Clone the repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/your-username/chatgpt4-method-prediction.git
cd chatgpt4-method-prediction
```
### Step 2: Install dependencies
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
### Step 3: Add your API Key
Replace the placeholder "YOUR_OPENAI_API_KEY_HERE" in the main.py file with your actual OpenAI API key:
```bash
api_key = "YOUR_OPENAI_API_KEY_HERE"
```
### Step 4: Prepare the CSV file
Ensure you have a CSV file (e.g., input_prompts.csv) with a column named prompt. Each row in this column should contain a prompt you want to generate a method name for.
### Step 5: Run the script
Once everything is set up, run the main script to process the CSV file and generate outputs:
```bash
python main.py
```
# Requirements
See requirements.txt for the full list of dependencies. Major dependencies include:

* openai: SDK for interacting with the ChatGPT-4 API.
* pandas: For reading and writing CSV files.
