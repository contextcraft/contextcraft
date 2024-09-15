# CodeBERT Method Name Prediction

This project fine-tunes the CodeBERT model to predict method names from functional descriptions for both Java and Python datasets. It uses Hugging Face's `transformers` library for model fine-tuning and testing.

## Project Structure

```bash
codebert_project/
│
├── data_load.py            # Module to load and preprocess the data
├── CodeBERT_FineTune.py    # Module to fine-tune the CodeBERT model
├── CodeBERT_Test.py        # Module to test the fine-tuned CodeBERT model
├── main.py                 # Main script for Java dataset (java_train.csv, java_test.csv)
├── python_main.py          # Main script for Python dataset (python_train.csv, python_test.csv)
└── requirements.txt        # Required Python libraries

```
## Requirements
Python 3.7+: Ensure that you have Python installed.
Install the required libraries using pip: 
```bash
pip install -r requirements.txt
```
## How to run
```bash
python main.py
```
```bash
python python_main.py
```
