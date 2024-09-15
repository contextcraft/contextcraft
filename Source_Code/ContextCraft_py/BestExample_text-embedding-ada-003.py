import openai
import numpy as np
import pandas as pd

class BestExample:
    def __init__(self, csv_file_path, api_key, model_name="text-embedding-ada-003"):
        # Load CSV file
        self.df = pd.read_csv(csv_file_path)
        # Store the API key and model name
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key

    def get_sentence_embedding(self, text):
        """Generates embeddings for the input text using text-embedding-ada-003."""
        response = openai.Embedding.create(
            input=text,
            model=self.model_name
        )
        embedding = np.array(response['data'][0]['embedding'])
        return embedding

    def find_top_n_similar_descriptions(self, input_description, n=10):
        """Finds the top N semantically similar functional descriptions."""
        input_embedding = self.get_sentence_embedding(input_description)
        similarities = []

        # Compute similarity for each functional description in the DataFrame
        for index, row in self.df.iterrows():
            description = row['Functional Description']
            description_embedding = self.get_sentence_embedding(description)
            similarity = np.dot(input_embedding, description_embedding) / (np.linalg.norm(input_embedding) * np.linalg.norm(description_embedding))
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
    
    # Provide your OpenAI API key
    api_key = "your_openai_api_key"  # Replace with your actual OpenAI API key

    # Create an instance of BestExample with the CSV file and API key
    best_example = BestExample(csv_file_path, api_key)

    # Input functional description for comparison
    input_description = "retrieve the status of this object"  # Replace with your input description

    # Find top 10 similar functional descriptions
    top_similar = best_example.find_top_n_similar_descriptions(input_description, n=10)

    # Print the top 10 similar functional descriptions with their method names
    print(top_similar)
