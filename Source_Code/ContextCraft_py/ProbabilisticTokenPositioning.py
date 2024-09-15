import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from collections import defaultdict
import inflect

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
# class for PTP part
class ProbabilisticTokenPositioning:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.inflect_engine = inflect.engine()
        self.position_counts = defaultdict(lambda: defaultdict(int))
        self.token_occurrences = defaultdict(int)

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts."""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def extract_tokens(self, text):
        """Extract and lemmatize tokens from text."""
        tokens = text.lower().split()
        processed_tokens = []
        for token in tokens:
            singular_token = self.inflect_engine.singular_noun(token)
            if singular_token:
                token = singular_token
            base_form_token = self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token))
            processed_tokens.append(base_form_token)
        return processed_tokens

    def calculate_position_probabilities(self, df):
        """Calculate token position probabilities."""
        for _, row in df.iterrows():
            fd_tokens = self.extract_tokens(row['Functional Description'])
            mn_tokens = self.extract_tokens(row['Method Name'])
            for fd_token in fd_tokens:
                self.token_occurrences[fd_token] += 1
                if fd_token in mn_tokens:
                    index = mn_tokens.index(fd_token)
                    if index == 0:
                        self.position_counts[fd_token]['prefix'] += 1
                    elif index == len(mn_tokens) - 1:
                        self.position_counts[fd_token]['suffix'] += 1
                    else:
                        self.position_counts[fd_token]['infix'] += 1

        position_probabilities = defaultdict(dict)
        for token, positions in self.position_counts.items():
            total = self.token_occurrences[token]
            for position in ['prefix', 'infix', 'suffix']:
                position_probabilities[token][position] = positions[position] / total
        
        return position_probabilities

    def create_probability_df(self, position_probabilities):
        """Create DataFrame from position probabilities."""
        result_data = {
            'Token': [],
            'Prefix Probability': [],
            'Infix Probability': [],
            'Suffix Probability': []
        }
        for token, probabilities in position_probabilities.items():
            result_data['Token'].append(token)
            result_data['Prefix Probability'].append(probabilities.get('prefix', 0))
            result_data['Infix Probability'].append(probabilities.get('infix', 0))
            result_data['Suffix Probability'].append(probabilities.get('suffix', 0))

        return pd.DataFrame(result_data)
