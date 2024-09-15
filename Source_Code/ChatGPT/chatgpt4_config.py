# chatgpt4_config.py

import openai

class ChatGPT4Configuration:
    def __init__(self, api_key):
        """Initialize the configuration with the OpenAI API key."""
        self.api_key = api_key
        openai.api_key = self.api_key

    def get_chatgpt4_response(self, prompt):
        """Send a prompt to the ChatGPT-4 model and retrieve the response."""
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            return response.choices[0].message["content"].strip()
        except Exception as e:
            return f"Error: {str(e)}"
