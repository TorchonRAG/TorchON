import llama_index
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import (
    CSVReader, DocxReader, EpubReader, FlatReader, HTMLTagReader, HWPReader,
    IPYNBReader, ImageCaptionReader, ImageReader, ImageTabularChartReader,
    ImageVisionLLMReader, MarkdownReader, MboxReader, PDFReader, PagedCSVReader,
    PandasCSVReader, PptxReader, PyMuPDFReader, RTFReader, UnstructuredReader,
    VideoAudioReader, XMLReader
)
from llama_index.readers.chroma import ChromaReader
from llama_index.readers.web import (
    AsyncWebPageReader, BeautifulSoupWebReader, KnowledgeBaseWebReader,
    MainContentExtractorReader, NewsArticleReader, ReadabilityWebPageReader,
    RssNewsReader, RssReader, SimpleWebPageReader, SitemapReader,
    TrafilaturaWebReader, UnstructuredURLLoader, WholeSiteReader
)
import llama_parse
from llama_parse import LlamaParse

from langchain_core.documents.base import Document

import dspy
from dspy.modules.anthropic import Claude
from dspy.signatures import (
    ExplainTask,
    GenerateFieldDescription,
    GenerateInputFieldsData,
    GenerateOutputFieldsData,
    GetFeedbackOnGeneration,
    UnderstandTask,
    UpdateTaskDescriptionBasedOnFeedback,
)
from dspy.experimental.synthesizer.config import SynthesizerArguments
from dspy.experimental.synthesizer.instruction_suffixes import (
    INPUT_GENERATION_TASK_WITH_EXAMPLES_SUFFIX,
    INPUT_GENERATION_TASK_WITH_FEEDBACK_SUFFIX,
)
from dspy.signatures import (
    ExplainTask,
    GenerateFieldDescription,
    GenerateInputFieldsData,
    GenerateOutputFieldsData,
    GetFeedbackOnGeneration,
    UnderstandTask,
    UpdateTaskDescriptionBasedOnFeedback,
)
from dspy.utils import format_examples
from dspy.retrieve.chromadb_rm import ChromadbRM
from dspy.evaluate import Evaluate
# from .datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune
from dsp.modules.lm import LM
from dsp.utils.utils import deduplicate


import os
import random
from pathlib import Path
from typing import Any, List, Dict
import base64

import chromadb
from chromadb.utils import embedding_functions
from functools import wraps
from dotenv import load_dotenv, set_key
from pydantic import BaseModel
import gradio as gr
from src.utils.conversation import Conversation, register_conv_template, get_conv_template, SeparatorStyle, conv_templates, setup_conversations
from src.utils.gradio_utils import cancel_outputing, delete_last_conversation, wrap_gen_fn, reset_state, reset_textbox, State, shared_state, transfer_input

load_dotenv()
class DataProcessor:
    def __init__(self, source_file: str, collection_name: str, persist_directory: str):
        self.source_file = source_file
        self.collection_name = collection_name
        self.persist_directory = persist_directory

    def load_data_from_source_and_store(self) -> Any:
    # def load_data_from_source_and_store(source: Union[str, dict], collection_name: str, persist_directory: str) -> Any:
        """
        Loads data from various sources and stores the data in ChromaDB.

        :param source: A string representing a file path or a URL, or a dictionary specifying web content to fetch.
        print("Data loaded and stored successfully in ChromaDB.")
        """
        # Determine the file extension
        if isinstance(self.source_file, str):
            ext = os.path.splitext(self.source_file)[-1].lower()
        else:
            raise TypeError("Source must be a string (file path or URL).")

        # Load data using appropriate reader
        if ext == '.csv':
            reader = CSVReader()
        elif ext == '.docx':
            reader = DocxReader()
        elif ext == '.epub':
            reader = EpubReader()
        elif ext == '.html':
            reader = HTMLTagReader()
        elif ext == '.hwp':
            reader = HWPReader()
        elif ext == '.ipynb':
            reader = IPYNBReader()
        elif ext in ['.png', '.jpg', '.jpeg']:
            reader = ImageReader()  # Assuming ImageReader can handle common image formats
        elif ext == '.md':
            reader = MarkdownReader()
        elif ext == '.mbox':
            reader = MboxReader()
        elif ext == '.pdf':
            reader = PDFReader()
        elif ext == '.pptx':
            reader = PptxReader()
        elif ext == '.rtf':
            reader = RTFReader()
        elif ext == '.xml':
            reader = XMLReader()
        elif self.source_file.startswith('http'):
            reader = AsyncWebPageReader()  # Simplified assumption for URLs
        else:
            raise ValueError(f"Unsupported source type: {self.source_file}")

        # Use the reader to load data
        # data = reader.read(self.source_file)
        data = reader.load_data(self.source_file)  
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(name=self.collection_name)
        collection.add(
                documents=[i.text for i in data], # the text fields
                metadatas=[i.extra_info for i in data], # the metadata
                ids=[i.doc_id for i in data], # the generated ids
        )
        # retriever_model = ChromadbRM(collection_name, persist_directory)
        # retriever_model(data)

        return data
 
class APIKeyManager:
    @staticmethod
    def set_api_keys(anthropic_api_key: str, openai_api_key: str, github_api_key: str, huggingface_api_key: str):
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
        set_key(env_path, "HUGGINGFACE_API_KEY", huggingface_api_key)
        
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

def choose_reader(file_path: str) -> Optional[object]:
    """Selects the appropriate reader for a given file based on its extension."""
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    readers = {
        '.csv': CSVReader,
        '.docx': DocxReader,
        '.epub': EpubReader,
        '.html': HTMLTagReader,  # Assuming HTMLTagReader is for .html files
        '.hwp': HWPReader,
        '.ipynb': IPYNBReader,
        '.jpg': ImageReader,
        '.jpeg': ImageReader,
        '.png': ImageReader,
        '.bmp': ImageReader,
        '.tiff': ImageReader,
        '.gif': ImageReader,  # Assuming ImageReader can handle .gif
        '.md': MarkdownReader,
        '.mbox': MboxReader,
        '.pdf': PDFReader,
        '.pptx': PptxReader,
        '.rtf': RTFReader,
        '.xml': XMLReader,
        '.txt': FlatReader,  # Assuming FlatReader is for .txt files
        # .csv extension has multiple readers, I'm assuming PagedCSVReader and PandasCSVReader
        # are specific cases that would be handled elsewhere, hence using CSVReader as default
    }
    
    # For image files that have special readers
    image_readers = {
        '.jpg': ImageCaptionReader,  # or ImageTabularChartReader, ImageVisionLLMReader based on content
        '.jpeg': ImageCaptionReader,
        '.png': ImageTabularChartReader,
        # Add more mappings if there are specific readers for different image types
    }
    
    # If the file is an image and has a specialized reader, use that.
    if file_extension in image_readers:
        return image_readers[file_extension]()
    
    reader_class = readers.get(file_extension)
    return reader_class() if reader_class else None

class DocumentLoader:

    @staticmethod
    def load_documents_from_folder(folder_path: str) -> List[Document]:
        """Loads documents from files within a specified folder"""
        folder_path = "./add_your_files_here"
        documents = []
        for root, _, filenames in os.walk(folder_path):
            for filename in filenames:
                full_path = os.path.join(root, filename)
                
                reader = choose_reader(full_path)

                if reader:
                    print(f"Loading document from '{filename}' with {type(reader).__name__}")
                    
                    try:
                        docs = list(reader.load_data(input_files=[full_path]))
                        documents.extend(docs)
                        
                    except Exception as e:
                        print(f"Failed to load document from '{filename}'. Error: {e}")
        # Convert to langchain format
        documents = [ doc.to_langchain_format()
        for doc in documents
        ]                       
        return documents

class Upsert:
    def __init__(self, db_path ='./vectorstore', data_path='./your_files_here'):
        """
        Initialize the Upsert process with the path to the data and the database.
        """
        self.data_path = data_path
        self.db_path = db_path
        self.db = self._init_db()
        self.documents = self._load_documents()
        self.text_parser = SentenceSplitter(chunk_size=1024)

    def _init_db(self):
        """
        Initialize the ChromaDB client.
        """
        return chromadb.PersistentClient(path=self.db_path)

    def _load_documents(self):
        """
        Load documents from the specified directory.
        """
        return DocumentLoader.load_documents_from_folder(self.data_path)
    def process_and_upsert(self):
        """
        Process the documents and upsert them into the ChromaDB.
        """
        text_chunks, doc_idxs, doc_metadata, chunk_ids = self._process_documents()
        vector_store = self._upsert_to_db(text_chunks, doc_metadata, chunk_ids)
        return vector_store

    def _process_documents(self):
        """
        Process documents to split them into chunks and prepare metadata.
        """
        text_chunks = []
        doc_idxs = []
        doc_metadata = []
        chunk_ids = []

        for doc_idx, doc in enumerate(self.documents):
            cur_text_chunks = self.text_parser.split_text(doc.text)
            text_chunks.extend(cur_text_chunks)

            for chunk_idx, _ in enumerate(cur_text_chunks):
                doc_idxs.append(doc_idx)
                doc_metadata.append(doc.metadata)
                chunk_ids.append(f"{doc_idx}_{chunk_idx}")

        return text_chunks, doc_idxs, doc_metadata, chunk_ids

    def _upsert_to_db(self, text_chunks, doc_metadata, chunk_ids):
        """
        Upsert processed documents into the ChromaDB.
        """
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        vector_store = self.db.get_or_create_collection(name="TorchON", embedding_function=default_ef)
        vector_store.add(ids=chunk_ids, documents=text_chunks, metadatas=doc_metadata)
        return vector_store

# # Example usage:
# if __name__ == "__main__":
#     upserter = Upsert(data_path="./data", db_path="./chroma_db")
#     vector_store = upserter.process_and_upsert()
#     print("Upsert completed.")
class DescriptionSignature(dspy.Signature):
    """Write a simple search query that will help answer a complex question.
    https://github.com/stanfordnlp/dspy?tab=readme-ov-file#4-two-powerful-concepts-signatures--teleprompters
    """

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class SyntheticDataGenerator:
    def __init__(self, schema_class: Optional[BaseModel] = None, examples: Optional[List[dspy.Example]] = None):
        self.schema_class = schema_class
        self.examples = examples
        print("SyntheticDataGenerator initialized.")

    def generate(self, sample_size: int) -> List[dspy.Example]:
        print(f"Starting data generation for sample size: {sample_size}")
        if not self.schema_class and not self.examples:
            raise ValueError("Either a schema_class or examples must be provided.")
        if self.examples and len(self.examples) >= sample_size:
            print("No additional data generation needed.")
            return self.examples[:sample_size]

        additional_samples_needed = sample_size - (len(self.examples) if self.examples else 0)
        print(f"Generating {additional_samples_needed} additional samples.")
        generated_examples = self._generate_additional_examples(additional_samples_needed)

        return self.examples + generated_examples if self.examples else generated_examples

    def _define_or_infer_fields(self):
        print("Defining or inferring fields for data generation.")
        if self.schema_class:
            data_schema = self.schema_class.model_json_schema()
            properties = data_schema['properties']
        elif self.examples:
            inferred_schema = self.examples[0].__dict__['_store']
            descriptor = dspy.Predict(DescriptionSignature)
            properties = {field: {'description': str((descriptor(field_name=field, example=str(inferred_schema[field]))).description)}
                          for field in inferred_schema.keys()}
        else:
            properties = {}
        return properties

    def _generate_additional_examples(self, additional_samples_needed: int) -> List[dspy.Example]:
        print(f"Generating {additional_samples_needed} additional examples.")
        properties = self._define_or_infer_fields()
        class_name = f"{self.schema_class.__name__ if self.schema_class else 'Inferred'}Signature"
        fields = self._prepare_fields(properties)

        signature_class = type(class_name, (dspy.Signature,), fields)
        generator = dspy.Predict(signature_class, n=additional_samples_needed)
        response = generator(sindex=str(random.randint(1, additional_samples_needed)))

        return [dspy.Example({field_name: getattr(completion, field_name) for field_name in properties.keys()})
                for completion in response.completions]

    def _prepare_fields(self, properties) -> dict:
        print("Preparing fields for the signature class.")
        return {
            '__doc__': f"Generates the following outputs: {{{', '.join(properties.keys())}}}.",
            'sindex': dspy.InputField(desc="a random string"),
            **{field_name: dspy.OutputField(desc=properties[field_name].get('description', 'No description'))
               for field_name in properties.keys()},
        }

class ClaudeModelManager:
    def __init__(self, model: str = "claude-3-opus-20240229", api_key: Optional[str] = None, api_base: Optional[str] = None, **kwargs):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.api_base = api_base
        self.kwargs = kwargs
        self.history = []  # Track request/response history if needed
        self.initialize_claude(**kwargs)

    def initialize_claude(self, **kwargs):
        print("Initializing Claude...")
        try:
            from anthropic import Anthropic, RateLimitError
            print("Successfully imported Anthropics's API client.")
        except ImportError as err:
            print("Failed to import Anthropics's API client.")
            raise ImportError("Claude requires `pip install anthropic`.") from err

        if not self.api_key:
            raise ValueError("API key is not set. Please ensure it's provided or set in the environment variables.")

        self.client = Anthropic(api_key=self.api_key, model=self.model, **kwargs)
        print("Anthropic client initialized with model:", self.model)

# # Example usage
# kwargs = {
#     "temperature": 0.7,
#     "max_tokens": 2048,
#     "top_p": 0.9,
#     "top_k": 40,
#     "n": 1,  # Number of generations
#     # Add other parameters as needed
# }
# claude_manager = ClaudeModelManager(api_key="ANTHROPIC_API_KEY", **kwargs)    
class ClaudeGenerate(self, **kwargs):
    def __init__(model="claude-3-opus-20240229", api_key="ANTHROPIC_API_KEY"):
        self.model = model
        self.api_key = api_key
        self.provider = "default"

        self.base_url = "https://api.anthropic.com/v1/messages"

    def basic_request(self, prompt: str, **kwargs):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2023-12-15",
            "content-type": "application/json"
            }

        data = {
            **kwargs,
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        print("Generating Using Claude...")
        response = requests.post(self.base_url, headers=headers, json=data)
        response = response.json()

        self.history.append({
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
        })
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.request(prompt, **kwargs)
        completions = [result["text"] for result in response["content"]]

        return completions

 

class SyntheticDataHandler:
    def __init__(self, examples: Optional[List[dspy.Example]] = None):
        self.generator = SyntheticDataGenerator(examples=examples)

    def generate_data(self, sample_size: int):
        return self.generator.generate(sample_size=sample_size)

    def configure_dspy_settings(lm_model):
        dspy.settings.configure(rm=colbertv2, lm=lm_model)

    def prepare_datasets(dataset):
        trainset = [x.with_inputs('question') for x in dataset.train]
        devset = [x.with_inputs('question') for x in dataset.dev]
        testset = [x.with_inputs('question') for x in dataset.test]
        return trainset, devset, testset

# class ModelCompilationAndEnsemble:
#     @staticmethod
#     def compile_or_load_models(recompile, trainset, num_models=4):
#         ensemble = []
#         if recompile:
#             metric_EM = dspy.evaluate.answer_exact_match
#             tp = BootstrapFewShotWithRandomSearch(metric=metric_EM, max_bootstrapped_demos=2, num_threads=NUM_THREADS)
#             claude_bs = tp.compile(Claude(), trainset=trainset[:50], valset=trainset[50:200])
#             ensemble = [prog for *_, prog in claude_bs.candidate_programs[:num_models]]
#         else:
#             for idx in range(num_models):
#                 claude_model = Claude(model=f'multihop_claude3opus_{idx}.json')
#                 ensemble.append(claude_model)
#         return ensemble

class LongFormContent(dspy.Module):
    def __init__(self):
        self.prompt_to_outline = dspy.ChainOfThought(Question2BlogOutline)
        self.topic_to_paragraph = dspy.ChainOfThought(Topic2Paragraph)
        self.proof_reader = dspy.ChainOfThought(ProofReader)
        self.title_generator = dspy.ChainOfThought(TitleGenerator)
    
    def forward(self, prompt):
        contexts = dspy.Retrieve(k=5)(prompt).passages
        contexts = "".join(contexts)
        raw_outline = self.prompt_to_outline(question=prompt, context=contexts).blog_outline
        outline = raw_outline.split(",")  # Add type hint in expanded Signature
        content = ""
        for topic in outline:
            topic_contexts = dspy.Retrieve(k=5)(topic).passages
            topic_contexts = "".join(topic_contexts)
            content += self.topic_to_paragraph(topic=topic, contexts=topic_contexts).paragraph
            content += "\n\n"
        content = self.proof_reader(blog_post=content).proofread_blog_post
        title = self.title_generator(blog_outline=raw_outline).title
        final_content = f"{title}\n\n{content}"
        return dspy.Prediction(blog=final_content)
class ChatbotManager:
    def __init__(self, model_name):
        self.model = model_name
        self.history = []

    def generate_response(self, text, top_p=0.9, temperature=0.7, repetition_penalty=1.0, max_length_tokens=2048, max_context_length_tokens=1024):


        kwargs = {
            "temperature": temperature,
            "max_tokens": max_length_tokens,
            "top_p": top_p,
            "top_k": 40,
            "n": 1,
            "repetition_penalty": repetition_penalty,
            "model": self.model,  
        }

        prompt = self.generate_prompt_with_history(text, self.history, max_length=max_context_length_tokens)
        if prompt is None:
            return "Error: Prompt generation failed.", self.history, "Generate: Failed"

        gradio_chatbot_output = ClaudeGenerate(kwargs=kwargs, prompt=prompt)
        self.history.append((text, gradio_chatbot_output))  

        return gradio_chatbot_output, self.history, "Generate: Success"

    def generate_prompt_with_history( text, history, max_length=2048):
        """
        Generate a prompt with history for the TorchON application.
        Args:
            text (str): The text prompt.
            history (list): List of previous conversation messages.
            max_length (int): The maximum length of the prompt.
        Returns:
            tuple: A tuple containing the generated prompt, conversation, and conversation copy. If the prompt could not be generated within the max_length limit, returns None.
        """
        user_role_ind = 0
        bot_role_ind = 1

        # Initialize conversation
        conversation = ClaudeGenerate(kwargs=kwargs, prompt=text)

        if history:
            conversation.messages = history

        # REVISE 👇🏻

        conversation = Conversation.append_message(conversation.roles[user_role_ind], text)
        conversation = Conversation.append_message(conversation.roles[bot_role_ind], "")

        # Create a copy of the conversation to avoid history truncation in the UI
        conversation_copy = conversation.copy()
        logger.info("=" * 80)
        logger.info(ChatbotManager.get_prompt(conversation))

        rounds = len(conversation.messages) // 2

        for _ in range(rounds):
            current_prompt = ChatbotManager.get_prompt(conversation)
            if len(conversation.messages) % 2 != 0:
                gr.Error("The messages between user and assistant are not paired.")
                return

            try:
                for _ in range(2):  # pop out two messages in a row
                    conversation.messages.pop(0)
            except IndexError:
                gr.Error("Input text processing failed, unable to respond in this round.")
                return None

        gr.Error("Prompt could not be generated within max_length limit.")
        return None

    def to_gradio_chatbot(conv):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(conv.messages[conv.offset :]):
            if i % 2 == 0:
            else:
                ret[-1][-1] = msg
        return ret
    def to_gradio_history(conv):
        """Convert the conversation to gradio history state."""
        return conv.messages[conv.offset :]

    def get_prompt(conv) -> str:
        """Get the prompt for generation."""
        system_prompt = conv.system_template.format(system_message=conv.system_message)
        if conv.sep_style == SeparatorStyle.DeepSeek:
            seps = [conv.sep, conv.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(conv.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            return conv.get_prompt

    def predict(text, chatbot, history, top_p, temperature, repetition_penalty, max_length_tokens, max_context_length_tokens, model_select_dropdown,):
        """
        Function to predict the response based on the user's input and selected model.
        Parameters:
        user_text (str): The input text from the user.
        user_image (str): The input image from the user.
        chatbot (str): The chatbot's name.
        history (str): The history of the chat.
        top_p (float): The top-p parameter for the model.
        temperature (float): The temperature parameter for the model.
        max_length_tokens (int): The maximum length of tokens for the model.
        max_context_length_tokens (int): The maximum length of context tokens for the model.
        model_select_dropdown (str): The selected model from the dropdown.
        Returns:
        generator: A generator that yields the chatbot outputs, history, and status.
        """
        print("running the prediction function")
        conversation = generate_prompt_with_history(
            text,
            history,
            max_length=max_context_length_tokens,
        )
        prompts = convert_conversation_to_prompts(conversation)
        gradio_chatbot_output = to_gradio_chatbot(conversation)

        full_response = ""

        "Generating..."

        print("flushed result to gradio")


        yield gradio_chatbot_output, to_gradio_history(conversation), "Generate: Success"


    def retry(
        text,
        chatbot,
        history,
        top_p,
        temperature,
        repetition_penalty,
        max_length_tokens,
        max_context_length_tokens,
        model_select_dropdown,
    ):
        if len(history) == 0:
            yield (chatbot, history, "Empty context")
            return

        chatbot.pop()
        history.pop()
        text = history.pop()[-1]
        # if type(text) is tuple:
        #     text, image = text

        yield from predict(
            text,
            chatbot,
            history,
            top_p,
            temperature,
            repetition_penalty,
            max_length_tokens,
            max_context_length_tokens,
            model_select_dropdown,
        )

class Application:
    def __init__(self):
        self.api_key_manager = APIKeyManager()
        self.data_processor = DataProcessor(source_file="", collection_name="adapt-a-rag", persist_directory="/your_files_here")
        self.claude_model_manager = ClaudeModelManager()
        self.synthetic_data_handler = SyntheticDataHandler()
        self.chatbot_manager = ChatbotManager()

        # For publishing  - add these prompts to .env file, then load them accordingly !
        self.title = ""
        self.system_prompt = ""
        self.example = ""
        self.hf_token = ""
        
    def set_api_keys(self, anthropic_api_key, openai_api_key):
        return self.api_key_manager.set_api_keys(anthropic_api_key, openai_api_key)

    def handle_file_upload(self, uploaded_file):
        self.data_processor.source_file = uploaded_file.name
        loaded_data = self.data_processor.load_data_from_source_and_store()
        print("Data from {uploaded_file.name} loaded and stored successfully.")
        return loaded_data

    def handle_synthetic_data(self, schema_class_name, sample_size):  # schema_class_name
        synthetic_data = self.synthetic_data_handler.generate_data(sample_size=int(sample_size))
        synthetic_data_str = "\n".join([str(data) for data in synthetic_data])
        print ("Generated {sample_size} synthetic data items:\n{synthetic_data_str}")
        return synthetic_data

    def generate_content(prompt):
        content_generator = LongFormContent()
        result = content_generator(prompt)
        return result.blog

    def handle_chatbot_interaction(self, text, model_select, top_p, temperature, repetition_penalty, max_length_tokens, max_context_length_tokens):
        chatbot_response, history, status = self.chatbot_manager.generate_response(text, None, model_select, top_p, temperature, repetition_penalty, max_length_tokens, max_context_length_tokens)
        return chatbot_response

    def publish(self):
        title = strip_invalid_filename_characters(self.title, max_bytes=30) # find functions
        api = HfApi(token=self.hf_token)
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
        api.upload_file(#chromaDB,
        )
        api.upload_file(# requirements.txt
        )
        add_space_secret(
            new_space.repo_id, "HF_TOKEN", self.hf_token, token=self.hf_token
        )
        return f"Published to https://huggingface.co/spaces/{new_space.repo_id}"
    def main(self):
        with gr.Blocks() as demo:
            with gr.Accordion("API Keys", open=True) as api_keys_accordion:
                with gr.Row():
                    anthropic_api_key_input = gr.Textbox(label="Anthropic API Key", type="password")
                    openai_api_key_input = gr.Textbox(label="OpenAI API Key", type="password")
                    hf_token_input = gr.Textbox(type="password", label="Hugging Face Write Token")
                    github_token_input = gr.Textbox(type="password", label="GitHub Write Token")

                submit_button = gr.Button("Submit")
                confirmation_output = gr.Textbox(label="Confirmation", visible=False)

                submit_button.click(
                    fn=self.set_api_keys,
                    inputs=[anthropic_api_key_input, openai_api_key_input, hf_token_input, github_token_input],
                    outputs=confirmation_output
                )

            with gr.Accordion("Upload Data") as upload_data_accordion:
                file_upload = gr.File(label="Upload Data Files")  # add folder loader
                folder_upload = gr.Files(label="Upload Folder", directory=True)
                webpage_input = gr.Textbox(label="Web Page URL")
                github_input = gr.Textbox(label="GitHub Repository Link")
                
                file_upload_button = gr.Button("Process Data")
                validation_output = gr.JSON(label="Validation Output")
                    
                file_upload_button.click(
                    fn=self.handle_file_upload,
                    inputs=[file_upload, folder_upload, webpage_input, github_input],
                    outputs=validation_output
                )

            with gr.Accordion("Generate Synthetic Data") as generate_data_accordion:
                schema_input = gr.Textbox(label="Schema Class Name")
                sample_size_input = gr.Number(label="Sample Size", value=100)
                synthetic_data_button = gr.Button("Generate Synthetic Data")
                synthetic_data_output = gr.Textbox()

                synthetic_data_button.click(
                    fn=self.handle_synthetic_data,
                    inputs=[schema_input, sample_size_input],
                    outputs=synthetic_data_output
                )

            with gr.Accordion("Chatbot") as chatbot_accordion:
                text_input = gr.Textbox(label="Enter your question")
                model_select = gr.Dropdown(label="Select Model", choices=list(self.chatbot_manager.models.keys()))
                top_p_input = gr.Slider(label="Top-p", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
                temperature_input = gr.Slider(label="Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
                repetition_penalty_input = gr.Slider(label="Repetition Penalty", min_value=1.0, max_value=2.0, value=1.1, step=0.1)
                max_length_tokens_input = gr.Number(label="Max Length Tokens", value=2048)
                max_context_length_tokens_input = gr.Number(label="Max Context Length Tokens", value=2048)
                chatbot_output = gr.Chatbot(label="Chatbot Conversation")
                submit_button = gr.Button("Submit")

                submit_button.click(
                    fn=self.handle_chatbot_interaction, 
                    inputs=[text_input, model_select,
                            top_p_input, temperature_input,
                            repetition_penalty_input,
                            max_length_tokens_input,
                            max_context_length_tokens_input
                        ],    outputs=chatbot_output
                    )

            with gr.Accordion("Publish"):
                title_input = gr.Textbox(label="Title")
                system_prompt_input = gr.Textbox(label="System prompt", lines=3)
                example_input = gr.Textbox(label="Example", lines=2)
                hf_token_input = gr.Textbox(label="Hugging Face Token", type="password")
                publish_button = gr.Button("Publish")
                
                def update_publish_info(title, system_prompt, example, hf_token):
                    self.title = title
                    self.system_prompt = system_prompt
                    self.example = example
                    self.hf_token = hf_token
                    return self.publish()
                
                publish_button.click(
                    fn=update_publish_info,
                    inputs=[title_input, system_prompt_input, example_input, hf_token_input],
                    outputs=[]
                )

        demo.launch()

if __name__ == "__main__":
    app = Application()
    app.main()
    