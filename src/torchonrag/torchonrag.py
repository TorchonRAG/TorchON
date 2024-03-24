from src.dataloaders.dataloader import DataProcessor, DocumentLoader
import chromadb
from chromadb.utils import embedding_functions
import dspy
from dsp.modules.anthropic import Claude
from dsp.modules.lm import LM
from dsp.utils.utils import deduplicate
from dspy.retrieve.chromadb_rm import ChromadbRM
import re
from src.torchonllms.torchonrequestclients import ClaudeGenerate
from src.torchonlongform.longform import LongFormContent, PromptToExample, PromptToRetrieval , Retriever , LongFormQA, LongFormQAWithAssertions


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
#     upserter = Upsert(data_path="./data", db_path="./chroma_db")
#     vector_store = upserter.process_and_upsert()
#     print("Upsert completed.")

class MyRetriever:
    def __init__(self, api_key):
        default_ef = embedding_functions.DefaultEmbeddingFunction()
        self.retriever_model = ChromadbRM(collection_name='TorchON', persist_directory='./vectorstore',embedding_function=default_ef)
        llm = Claude(model="claude-3-opus-20240229")
        dspy.settings.configure(lm=llm, rm=self.retriever_model)
#       self.retrieve = Retrieve(k=3)
        self.Response = ClaudeGenerate()
        self.SystemPrompt =  dspy.ChainOfThought(Retriever)

    def retrieve_passages(self, query):
        return self.retrieve(query).passages

    def format_prompt(self, query, passages):
        passages_str = "\n\n".join(passages)
        formatted_prompt = re.sub(r'\{list_of_retrieved_text_chunks\}', passages_str, query)
        return formatted_prompt

    def generate_response(self, systemprompt, query):
        passages = self.retrieve_passages(query)
        formatted_prompt = self.format_prompt(query, passages)
        systemprompt = self.SystemPrompt(query)
        response = self.Response(systemprompt, formatted_prompt)
        return response   