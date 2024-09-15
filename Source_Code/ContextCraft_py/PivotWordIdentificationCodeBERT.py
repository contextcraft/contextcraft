# PivotWordIdentificationCodeBERT.py
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

class PivotWordIdentificationCodeBERT:
    def __init__(self, model_name="microsoft/codebert-base"):
        """Initialize the CodeBERT model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

    def identify_pivot_words(self, input_sentence, masked_word="[MASK]"):
        """Identify pivot words in a sentence using CodeBERT."""
        # Replace the pivot word with the masked token
        tokenized_input = self.tokenizer(input_sentence, return_tensors="pt")

        # Get token ids for the input sentence
        input_ids = tokenized_input["input_ids"]

        # Predict the output for the masked token using CodeBERT
        with torch.no_grad():
            outputs = self.model(input_ids)

        # Get predictions for masked words
        prediction_scores = outputs.logits

        # Get the most likely token
        predicted_token_ids = torch.argmax(prediction_scores, dim=-1)
        predicted_tokens = self.tokenizer.convert_ids_to_tokens(predicted_token_ids[0])

        # Return the predicted words at the masked positions
        return predicted_tokens

    def find_pivot_word_in_sentence(self, sentence, target_word):
        """Identify if a word in the sentence is a pivot word."""
        # Tokenize the sentence
        tokenized_sentence = self.tokenizer.tokenize(sentence)

        # Find the position of the target word
        try:
            word_position = tokenized_sentence.index(target_word)
        except ValueError:
            raise ValueError(f"Word '{target_word}' not found in sentence.")

        # Replace the target word with the masked token
        tokenized_sentence[word_position] = self.tokenizer.mask_token
        masked_sentence = self.tokenizer.convert_tokens_to_string(tokenized_sentence)

        # Identify the pivot word
        return self.identify_pivot_words(masked_sentence)
