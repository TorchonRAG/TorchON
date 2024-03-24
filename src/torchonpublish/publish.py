import huggingface_hub
from huggingface_hub import HfApi

class TorchonPublisher:
    def __init__(self, title, hf_token, anthropic_api_key):
        self.title = title
        self.hf_token = hf_token
        self.anthropic_api_key = anthropic_api_key
        self.api = HfApi()

    def publish(self):
        title = (self.title, max_bytes:=30) # find functions
        api = (token:=self.hf_token)
        new_space = api.create_repo(
            repo_id=f"tonic-ai-torchon-{title}",
            repo_type="space",
            exist_ok=True,
            private=False,
            space_sdk="gradio",
            token=self.hf_token,
        )
        api.upload_file(
            repo_id=new_space.repo_id,
            path_or_fileobj='/deploytorchon/apptemplate.py',
            path_in_repo='app.py',
            token=self.hf_token,
            repo_type="space",
        )
        api.upload_file(
            repo_id=new_space.repo_id,
            path_or_fileobj='/deploytorchon/README_template.md',
            path_in_repo='README.md',
            token=self.hf_token,
            repo_type="space",
        )
        api.upload_file(
            repo_id=new_space.repo_id,
            path_or_fileobj='/vectorstore', #chromaDB,
            path_in_repo='/vectorstore', #chromadb
            token=self.hf_token,
            repo_type="space",
        )
        api.upload_file(
            repo_id=new_space.repo_id,
            path_or_fileobj='/deploytorchon/requirements_template.txt',
            path_in_repo='requirements.txt',
            token=self.hf_token,
            repo_type="space",
        )
        api.upload_folder(
            repo_id=new_space.repo_id,
            folder_path='./vectorstore',
            repo_type="space",
        )
        api.add_space_secret(
            new_space.repo_id, "HF_TOKEN", self.hf_token, token=self.hf_token
        )
        api.add_space_secret(
            new_space.repo_id, "ANTHROPIC_API_KEY", self.anthropic_api_key, token=self.anthropic_api_key
        )
        api.add_space_secret(
            new_space.repo_id, "SYSTEM_PROMPT", # FIX THIS PART self.anthropic_api_key, token=self.anthropic_api_key
        )
        return f"Published to https://huggingface.co/spaces/{new_space.repo_id}"
