# PivotWordIdentificationAda003.py
import openai
import numpy as np

class PivotWordIdentificationAda003:
    def __init__(self, api_key, model_name="text-embedding-ada-003"):
        """Initialize the Ada embedding model and set up the API key."""
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key

    def get_embedding(self, text):
        """Get the embedding of a given text using OpenAI's Ada model."""
        response = openai.Embedding.create(input=text, model=self.model_name)
        return np.array(response['data'][0]['embedding'])

    def identify_pivot_words(self, sentence, target_word, candidate_words):
        """
        Identify the pivot word by calculating the semantic similarity
        between the masked sentence and a list of candidate words.
        """
        # Tokenize the sentence and replace the target word with a placeholder
        tokenized_sentence = sentence.split()
        try:
            word_position = tokenized_sentence.index(target_word)
        except ValueError:
            raise ValueError(f"Word '{target_word}' not found in sentence.")

        # Replace the target word with a placeholder
        tokenized_sentence[word_position] = "[MASK]"
        masked_sentence = " ".join(tokenized_sentence)

        # Get the embedding of the masked sentence
        masked_sentence_embedding = self.get_embedding(masked_sentence)

        # Get embeddings for candidate words
        candidate_embeddings = np.array([self.get_embedding(word) for word in candidate_words])

        # Calculate cosine similarity between the masked sentence and candidate words
        similarities = np.dot(candidate_embeddings, masked_sentence_embedding) / (
            np.linalg.norm(candidate_embeddings, axis=1) * np.linalg.norm(masked_sentence_embedding)
        )

        # Find the candidate word with the highest similarity score
        best_match_idx = np.argmax(similarities)
        best_match_word = candidate_words[best_match_idx]

        return best_match_word
