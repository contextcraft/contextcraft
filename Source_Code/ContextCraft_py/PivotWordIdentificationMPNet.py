# PivotWordIdentificationMPNet.py
from sentence_transformers import SentenceTransformer, util

class PivotWordIdentificationMPNet:
    def __init__(self, model_name="all-mpnet-base-v2"):
        """Initialize the All-MPNET-base-v2+ model."""
        self.model = SentenceTransformer(model_name)

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

        # Encode the masked sentence
        masked_sentence_embedding = self.model.encode(masked_sentence)

        # Encode the candidate words
        candidate_embeddings = self.model.encode(candidate_words)

        # Calculate similarity scores between the masked sentence and candidate words
        similarity_scores = util.cos_sim(masked_sentence_embedding, candidate_embeddings)

        # Find the candidate with the highest similarity score
        best_match_idx = similarity_scores.argmax().item()
        best_match_word = candidate_words[best_match_idx]

        return best_match_word
