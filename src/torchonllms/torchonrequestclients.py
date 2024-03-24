# ./src/torchonrequestclients/torchonrequestclients.py

import requests

class ClaudeGenerate:
    def __init__(self, systemprompt = None, prompt = None, model="claude-3-opus-20240229", api_key="ANTHROPIC_API_KEY"):
        self.model = model
        self.api_key = api_key
        self.provider = "default"
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.systemprompt = systemprompt
        self.prompt = prompt

    def basic_request(self, systemprompt: str, prompt: str, **kwargs):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2023-12-15",
            "content-type": "application/json"
        }

        data = {
            **kwargs,
            "model": self.model,
            "system": self.systemprompt,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        print("Generating Using Claude...")
        response = requests.post(self.base_url, headers=headers, json=data)
        response = response.json()

        # self.history.append({
        #     "prompt": prompt,
        #     "response": response,
        #     "kwargs": kwargs,
        # })
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.basic_request(self.systemprompt, prompt, **kwargs)
        completions = [result["text"] for result in response["content"]]

        return completions    
