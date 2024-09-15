# Llama3 Method Name Prediction

This project predicts method names based on input prompts using the Llama3 API. It reads prompts from a CSV file, generates method names or other relevant predictions, and saves the outputs to a new CSV file.

## Project Structure

The project is organized into several modules, with a clear separation of concerns:

```bash
llama3_method_prediction
main.py # Main script to run the process
llama3_config.py # Contains Llama3Configuration class
text_processing.py # Contains TextProcessing class
csv_handler.py # Handles reading and writing CSV files
method_name_pred.py # Orchestrates the process using the above classes
```

### Project Modules

- **Llama3Configuration**: Handles setting up the Llama3 API client and authenticating with the API key.
- **TextProcessing**: Manages input prompts and communicates with the Llama3 API to generate responses.
- **CSVHandler**: Reads prompts from a CSV file and writes generated outputs back to a CSV.
- **MethodNamePrediction**: Coordinates the overall process, connecting the Llama3 API, CSV handling, and text processing.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- A valid Llama3 API key

### Step 1: Clone the repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/your-username/llama3-method-prediction.git
cd llama3-method-prediction
```
### Step 2: Install dependencies
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```
### Step 3: Add your API Key
Replace the placeholder "YOUR_API_KEY_HERE" in the main.py file with your actual Llama3 API key:
```bash
api_key = "YOUR_API_KEY_HERE"
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

** pandas: For reading and writing CSV files.
** groq: SDK for interacting with the Llama3 API.
