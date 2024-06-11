import pandas as pd
from ProbabilisticTokenPositioning import ProbabilisticTokenPositioning
from PivotWordIdentification import PivotWordIdentification
from LLMBasedFeedback import LLMBasedFeedback
from BestExample import BestExample

class ContextCraft:
    def __init__(self, csv_file_path, api_client):
        self.df = pd.read_csv(csv_file_path)
        self.prob_token_positioning = ProbabilisticTokenPositioning()
        self.pivot_word_id = PivotWordIdentification()
        self.llm_feedback = LLMBasedFeedback(api_client)
        self.best_example = BestExample(csv_file_path)
        self.probability_df = None

    def process_csv_file(self):
        """Process the CSV file using the ProbabilisticTokenPositioning, PivotWordIdentification, and LLMBasedFeedback classes."""
        # Calculate position probabilities
        position_probabilities = self.prob_token_positioning.calculate_position_probabilities(self.df)
        self.probability_df = self.prob_token_positioning.create_probability_df(position_probabilities)

        # Process each row in the DataFrame
        self.df['Example Description'] = ''

        for index, row in self.df.iterrows():
            func_desc = row['Functional Description']
            method_name = row['Method Name']

            # Get prefix, infix, and suffix probabilities
            tokens = self.prob_token_positioning.extract_tokens(func_desc)
            prefix, infix, suffix = self.get_token_positions(tokens)

            # Get pivot words
            pivot_words = self.pivot_word_id.find_best_description_tokens(method_name, func_desc)
            pivot_words_str = self.format_pivot_words(pivot_words)

            # Get LLM feedback
            predicted_name = self.llm_feedback.predict_method_name(func_desc)
            feedback = self.llm_feedback.generate_feedback(predicted_name, method_name)

            # Format example description
            example_desc = self.format_example_description(
                func_desc, method_name, prefix, infix, suffix, pivot_words_str, feedback
            )
            self.df.at[index, 'Example Description'] = example_desc

        # Save processed DataFrame to a new CSV file
        self.df.to_csv('processed_data.csv', index=False)

    def get_token_positions(self, tokens):
        """Determine the most likely prefix, infix, and suffix for tokens."""
        prefix = None
        infix = None
        suffix = None
        max_prefix_prob = max_infix_prob = max_suffix_prob = 0

        for token in tokens:
            if token in self.probability_df['Token'].values:
                row = self.probability_df[self.probability_df['Token'] == token]
                prefix_prob = row['Prefix Probability'].values[0]
                infix_prob = row['Infix Probability'].values[0]
                suffix_prob = row['Suffix Probability'].values[0]

                if prefix_prob > max_prefix_prob:
                    max_prefix_prob = prefix_prob
                    prefix = token
                if infix_prob > max_infix_prob:
                    max_infix_prob = infix_prob
                    infix = token
                if suffix_prob > max_suffix_prob:
                    max_suffix_prob = suffix_prob
                    suffix = token

        return (prefix, max_prefix_prob), (infix, max_infix_prob), (suffix, max_suffix_prob)

    def format_pivot_words(self, pivot_words):
        """Format pivot words into a string for the example description."""
        return ", ".join(
            [f"'{k}': ('{v[0]}', {v[1] * 100:.2f}%)" for k, v in pivot_words.items()]
        )

    def format_example_description(self, func_desc, method_name, prefix, infix, suffix, pivot_words, feedback):
        """Format the example description according to the specified style."""
        prefix_str = f"Infix: ('{infix[0]}', Probability: {infix[1] * 100:.2f}%)" if infix[0] else ""
        suffix_str = f"Suffix: ('{suffix[0]}', Probability: {suffix[1] * 100:.2f}%)" if suffix[0] else ""
        context = f"In this example, the most likely word are: {prefix_str} {suffix_str}."
        return (
            f"Functional Description: \"{func_desc}\"\n"
            f"Method Name: '{method_name}'\n"
            f"Context:\n"
            f"• {context}\n"
            f"• The semantic similarity between tokens of description and method name: {pivot_words}.\n"
            f"• {feedback}"
        )

    def find_similar_examples(self, input_description):
        """Find the top 10 similar functional descriptions."""
        top_similar = self.best_example.find_top_n_similar_descriptions(input_description, n=10)

        # Initialize results DataFrame
        results = top_similar.copy()
        results['Example Description'] = ''

        for index, row in results.iterrows():
            func_desc = row['Functional Description']
            method_name = row['Method Name']

            # Get prefix, infix, and suffix probabilities
            tokens = self.prob_token_positioning.extract_tokens(func_desc)
            prefix, infix, suffix = self.get_token_positions(tokens)

            # Get pivot words
            pivot_words = self.pivot_word_id.find_best_description_tokens(method_name, func_desc)
            pivot_words_str = self.format_pivot_words(pivot_words)

            # Get LLM feedback
            predicted_name = self.llm_feedback.predict_method_name(func_desc)
            feedback = self.llm_feedback.generate_feedback(predicted_name, method_name)

            # Format example description
            example_desc = self.format_example_description(
                func_desc, method_name, prefix, infix, suffix, pivot_words_str, feedback
            )
            results.at[index, 'Example Description'] = example_desc

        return results

# Example Usage
if __name__ == "__main__":
    # Path to the CSV file containing functional descriptions and method names
    csv_file_path = 'path_to_file.csv'  # Replace with CSV file path
    
    
    api_client = LLMAPIClient()
    
    # Create an instance of ContextCraft with the CSV file and the LLM API client
    context_craft = ContextCraft(csv_file_path, api_client)
    
    # Process the CSV file to generate and save results
    context_craft.process_csv_file()

    # Input functional description for comparison
    input_description = "retrieve the status of this object"  # input description

    # Find and process the top 10 similar functional descriptions
    similar_examples = context_craft.find_similar_examples(input_description)
    
    # Print the results
    print(similar_examples['Example Description'].to_string(index=False))

