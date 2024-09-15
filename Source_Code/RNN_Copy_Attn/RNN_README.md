# RNN Copy Attention Model for Method Name Prediction

This project fine-tunes an RNN model with an attention mechanism to predict method names from functional descriptions in Python datasets. The model uses TensorFlow and Hugging Face's transformers for tokenization and training.

## Project Structure

```bash
rnn_copy_attention_project/
│
├── data_preprocessor.py    # Module for data loading and preprocessing
├── RNN_Copy_Attn.py        # Module to define the RNN Copy Attention model
├── ShapeValidator.py       # Module for shape validation utilities
├── trainer.py              # Module to train the model
├── evaluator.py            # Module to evaluate the model
├── requirements.txt        # Required Python libraries
└── main.py                 # Main script to run the training and testing
```
## Requirements
Make sure you have Python 3.7+ installed. You can install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```
## How to Run
```bash
python main.py
```
