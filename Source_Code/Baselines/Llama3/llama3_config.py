# llama3_config.py

from groq import Groq

class Llama3Configuration:
    def __init__(self, api_key):
        """Initialize the configuration with the Llama3 API key."""
        self.api_key = api_key
        self.client = None

    def configure_client(self):
        """Configure the client to interact with the Llama3 API."""
        self.client = Groq(api_key=self.api_key)
    
    def get_client(self):
        """Returns the configured client for interacting with the Llama3 API."""
        if self.client is None:
            raise ValueError("Client not configured. Call configure_client() first.")
        return self.client
