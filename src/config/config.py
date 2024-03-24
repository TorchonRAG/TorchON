# ./src/config/config.py

from dotenv import load_dotenv
from pathlib import Path
import os
load_dotenv()

class APIKeyManager:
    @staticmethod
    def set_api_keys(anthropic_api_key: str, openai_api_key: str, github_api_key: str, hf_token: str):
        """
        Function to securely set API keys by updating the .env file in the application's directory.
        """
        print("Setting API keys...")
        env_path = Path('.') / '.env'
        
        print(f"Loading existing .env file from: {env_path}")
        load_dotenv(dotenv_path=env_path, override=True)
        
        print("Updating .env file with new API keys...")
        set_key(env_path, "ANTHROPIC_API_KEY", anthropic_api_key)
        set_key(env_path, "OPENAI_API_KEY", openai_api_key)
        set_key(env_path, "GITHUB_API_KEY", github_api_key)
        set_key(env_path, "HUGGINGFACE_API_KEY", hf_token)
        
        print("API keys updated successfully.")
        return "API keys updated successfully in .env file. Please proceed with your operations."

    @staticmethod
    def set_prompts(field_prompt: str, example_prompt: str, example_prompt2: str, title_prompt: str, description_prompt: str, system_prompt: str):
        """
        Function to securely set various prompts by updating the .env file in the application's directory.
        """
        print("Setting prompts...")
        env_path = Path('.') / '.env'
        
        print(f"Loading existing .env file from: {env_path}")
        load_dotenv(dotenv_path=env_path, override=True)
        
        print("Updating .env file with new prompts...")
        set_key(env_path, "FIELDPROMPT", field_prompt)
        set_key(env_path, "EXAMPLEPROMPT", example_prompt)
        set_key(env_path, "EXAMPLE_PROMPT2", example_prompt2)
        set_key(env_path, "TITLE_PROMPT", title_prompt)
        set_key(env_path, "DESCRIPTIONPROMPT", description_prompt)
        set_key(env_path, "SYSTEM_PROMPT", system_prompt)
        
        print("Prompts updated successfully.")
        return "Prompts updated successfully in .env file. Please proceed with your operations."

    @staticmethod
    def load_api_keys_and_prompts():
        """
        Loads API keys and prompts from an existing .env file into the application's environment.
        """
        print("Loading API keys and prompts...")
        env_path = Path('.') / '.env'
        
        print(f"Loading .env file from: {env_path}")
        load_dotenv(dotenv_path=env_path)
        
        print("Accessing variables from the environment...")
        variables = {
            "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "GITHUB_API_KEY": os.getenv("GITHUB_API_KEY"),
            "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY"),
            "FIELDPROMPT": os.getenv("FIELDPROMPT"),
            "EXAMPLEPROMPT": os.getenv("EXAMPLEPROMPT"),
            "EXAMPLE_PROMPT2": os.getenv("EXAMPLE_PROMPT2"),
            "TITLE_PROMPT": os.getenv("TITLE_PROMPT"),
            "DESCRIPTIONPROMPT": os.getenv("DESCRIPTIONPROMPT"),
            "SYSTEM_PROMPT": os.getenv("SYSTEM_PROMPT")
        }
        
        print("API keys and prompts loaded successfully.")
        # Optionally, print a confirmation or return the loaded values
        return variables
    