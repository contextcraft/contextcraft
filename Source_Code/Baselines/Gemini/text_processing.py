# text_processing.py

class TextProcessing:
    def __init__(self, chat_session):
        self.chat_session = chat_session

    def generate_output_for_prompt(self, prompt):
        """Generates the response for the given prompt using the chat session."""
        response = self.chat_session.send_message(prompt)
        return response.text
