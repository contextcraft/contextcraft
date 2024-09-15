import torch
from transformers import BertModel, BertTokenizer
from scipy.spatial.distance import cosine
# class for PWI to identify pivot word identification
class PivotWordIdentification:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

    def get_token_embeddings(self, text):
        """Generates BERT embeddings for each token in the input text."""
        encoded_input = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            output = self.model(**encoded_input)
        embeddings = output.hidden_states[-1].squeeze(0)  # Get the embeddings from the last layer
        return embeddings, encoded_input['input_ids']

    def find_best_description_tokens(self, method_name, description):
        """Finds the description token with the highest similarity score for each method name token."""
        name_embeddings, name_ids = self.get_token_embeddings(method_name)
        desc_embeddings, desc_ids = self.get_token_embeddings(description)

        name_tokens = [self.tokenizer.decode([token_id]).strip() for token_id in name_ids[0]]
        desc_tokens = [self.tokenizer.decode([token_id]).strip() for token_id in desc_ids[0]]

        best_matches = {}
        for i, (name_embed, name_token) in enumerate(zip(name_embeddings, name_tokens)):
            if name_token in ['[CLS]', '[SEP]']:
                continue
            highest_score = -1
            best_token = None
            for j, (desc_embed, desc_token) in enumerate(zip(desc_embeddings, desc_tokens)):
                if desc_token in ['[CLS]', '[SEP]'] or desc_token.startswith('##'):
                    continue
                similarity = 1 - cosine(name_embed, desc_embed)
                if similarity > highest_score:
                    highest_score = similarity
                    best_token = desc_token
            best_matches[name_token] = (best_token, highest_score)

        return best_matches
