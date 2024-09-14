# text_processing.py

class TextProcessing:
    def __init__(self, client):
        self.client = client

    def generate_output_for_prompt(self, prompt):
        """Generates the response for a given prompt using the Llama3 client."""
        response = self.client.generate(prompt)
        return response['text'] if 'text' in response else "No response"
