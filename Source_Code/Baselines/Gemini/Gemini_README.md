# Method Name Generation with Google Generative AI

This project generates method names or outputs based on a list of prompts stored in a CSV file. It uses Google Generative AI's `gemini-1.0-pro` model to generate responses for each prompt. The results are saved back to a new CSV file with the generated output.

## Project Structure

The project is organized as follows:

```bash
method_name_generation
main.py # Main script to run the process
model_config.py # Contains the ModelConfiguration class
text_processing.py # Contains the TextProcessing class
csv_handler.py # Handles reading and writing CSV files
method_name_gen.py # Orchestrates the process using the above classes
```

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- A Google Generative AI API key

### Step 1: Clone the repository

Clone this project repository to your local machine.

```bash
git clone https://github.com/your-username/method-name-generation.git
cd method-name-generation
```
### Step 2: Install dependencies
Install the required dependencies using pip and the requirements.txt file.
```bash
pip install -r requirements.txt
```
### Step 3: Add your API Key
Replace the placeholder "YOUR_API_KEY_HERE" in the main.py file with your actual Google Generative AI API key.
```bash
api_key = "YOUR_API_KEY_HERE"
```
### Step 4: Prepare the CSV file
Make sure you have a CSV file (e.g., input_prompts.csv) with a column named prompt. Each row in this column should contain a prompt you want to generate a method name or output for.
### Step 5: Run the script
Once the setup is complete, run the main script to process the CSV file and generate outputs.
```bash
python main.py
```
