import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

class CodeBERT_Test:
    def __init__(self, model_path):
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base')

    def predict(self, functional_description):
        inputs = self.tokenizer(functional_description, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
        outputs = self.model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        return predictions.item()

    def evaluate(self, test_df):
        test_df['Predicted Method Name'] = test_df['Functional Description'].apply(self.predict)
        test_df['Correct Prediction'] = test_df['Method Name'] == test_df['Predicted Method Name']
        accuracy = test_df['Correct Prediction'].mean()
        return accuracy, test_df
