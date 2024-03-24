# ./src/torchonragmaker/torchonragmaker.py
import dspy
from src.torchonlongform.longform import LongFormContent, PromptToExample, PromptToRetrieval , Retriever , LongFormQA, LongFormQAWithAssertions

class CreateRAG:
    def __init__(self):
        self.prompt_to_retrieval_prompt = dspy.ChainOfThought(PromptToRetrieval) # add clarifai system prompt here?
        self.prompt_to_example_input = dspy.ChainOfThought(PromptToExample)

    def create_rag(self, question):
        retrieval_prompt = r"\{list_of_retrieved_text_chunks\}\n\n" + self.prompt_to_retrieval_prompt(question=question)
        example_input = self.prompt_to_example_input(question=question)
        return retrieval_prompt , example_input
    def __call__(self, question):
        return self.create_rag(question)
    