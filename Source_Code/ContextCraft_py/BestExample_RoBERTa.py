import torch
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import pandas as pd

class BestExample:
    def __init__(self, csv_file_path):
        # Load CSV file
        self.df = pd.read_csv(csv_file_path)
        # Load RoBERTa tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        self.model = AutoModel.from_pretrained('roberta-base', output_hidden_states=True)

    def get_sentence_embedding(self, text):
        """Generates RoBERTa embeddings for the input text."""
        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model(**encoded_input)
        # Get the mean of the embeddings from the last hidden state
        sentence_embedding = torch.mean(output.last_hidden_state, dim=1)
        return sentence_embedding

    def find_top_n_similar_descriptions(self, input_description, n=10):
        """Finds the top N semantically similar functional descriptions."""
        input_embedding = self.get_sentence_embedding(input_description)
        similarities = []

        # Compute similarity for each functional description in the DataFrame
        for index, row in self.df.iterrows():
            description = row['Functional Description']
            description_embedding = self.get_sentence_embedding(description)
            similarity = 1 - cosine(input_embedding.squeeze(), description_embedding.squeeze())
            similarities.append((index, similarity))

        # Sort by similarity score in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Extract top N similar descriptions
        top_n_indices = [index for index, similarity in similarities[:n]]
        top_n_descriptions = self.df.iloc[top_n_indices]

        return top_n_descriptions[['Functional Description', 'Method Name']]

# Example Usage
if __name__ == "__main__":
    # Path to the CSV file containing functional descriptions and method names
    csv_file_path = 'path_to_your_file.csv'  # Replace with your CSV file path
    
    # Create an instance of BestExample with the CSV file
    best_example = BestExample(csv_file_path)
    
    # Input functional description for comparison
    input_description = "retrieve the status of this object"  # Replace with your input description

    # Find top 10 similar functional descriptions
    top_similar = best_example.find_top_n_similar_descriptions(input_description, n=10)

    # Print the top 10 similar functional descriptions with their method names
    print(top_similar)
