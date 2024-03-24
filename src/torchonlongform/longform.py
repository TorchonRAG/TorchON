# ./src/torchonlongform/longform.py

import dspy

# from dspy.signatures import (
#     ExplainTask,
#     GenerateFieldDescription,
#     GenerateInputFieldsData,
#     GenerateOutputFieldsData,
#     GetFeedbackOnGeneration,
#     UnderstandTask,
#     UpdateTaskDescriptionBasedOnFeedback,
# )
# from dspy.experimental.synthesizer.config import SynthesizerArguments
# from dspy.experimental.synthesizer.instruction_suffixes import (
#     INPUT_GENERATION_TASK_WITH_EXAMPLES_SUFFIX,
#     INPUT_GENERATION_TASK_WITH_FEEDBACK_SUFFIX,
# )
# from dspy.signatures import (
#     ExplainTask,
#     GenerateFieldDescription,
#     GenerateInputFieldsData,
#     GenerateOutputFieldsData,
#     GetFeedbackOnGeneration,
#     UnderstandTask,
#     UpdateTaskDescriptionBasedOnFeedback,
# )
import regex as re
from dspy.predict import Retry
from dspy.datasets import HotPotQA

from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dsp.utils import EM, normalize_text
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from .utils import extract_text_by_citation, correct_citation_format, has_citations, citations_check
from dsp.utils.utils import deduplicate
# from dspy.retrieve.chromadb_rm import ChromadbRM
# from dspy.evaluate import Evaluate
# from .datasets.hotpotqa import HotPotQA
# from dspy.teleprompt import BootstrapFewShotWithRandomSearch, BootstrapFinetune
# from dsp.modules.lm import LM
# from dsp.utils.utils import deduplicate
class PromptToRetrieval(dspy.Signature):
    """
    you should produce a prompt that will be optimized for context answering with appropriate length, and field and topic specific awareness , discussing the implications of the content
    Your task is to produce a prompt that will be used in the following format : 
    {list_of_retrieved_text_chunks}\n {your_answer_here}
    the outcome is an infomational response based on the context provided
    """
    prompt = dspy.InputField()
    examples = dspy.OutputField(desc="A retrieval prompt designed to optimize the retrieval answer for the given topic and context")

class PromptToExample(dspy.Signature):
    """
    Your task is to convert a given prompt into an example input from a user for the given information retriever.
    """
    
    prompt = dspy.InputField()
    example = dspy.OutputField(desc="An example to demonstrate the information retrieval of this application.")

class Retriever(dspy.Signature):
    """
    Your task is to create the instruction system prompt to optimize for answering a question based on context and topic.
    the user will provide passages and facts and you should create a system prompt to answer his query optimized for field and contextual knowledge
    only produce a system prompt based on the given context and topic to produce the most informational and rich answer based on that field.
    """
    prompt = dspy.InputField()
    examples = dspy.OutputField(desc="A system prompt designed to optimize the retrieval answer for the given topic and context")

class Question2ReportOutline(dspy.Signature):
    """
    Your task is to write a report that will help answer the given question. 
    Use the contexts to evaluate the structure of the report.
    Optimize for informational content, audience interest and describe the implications of the contexts in relation to the question.
    """
    
    question = dspy.InputField()
    contexts = dspy.InputField()
    blog_outline = dspy.OutputField(desc="A comma separated list of topics.")

class Topic2Paragraph(dspy.Signature):
    """
    Your task is to write a paragraph that explains a topic based on the retrieved contexts.
    """
    
    topic = dspy.InputField(desc="A topic to write a paragraph about based on the information in the contexts.")
    contexts = dspy.InputField(desc="contains relevant information about the topic.")
    paragraph = dspy.OutputField()

class ProofReader(dspy.Signature):
    """
    Proofread a blog post and output a more well written version of the original post.
    """
    
    blog_post = dspy.InputField()
    proofread_blog_post = dspy.OutputField()    

class TitleGenerator(dspy.Signature):
    """
    Write a title for a blog post given a description of the topics the blog covers as input.
    """

    blog_outline = dspy.InputField()
    title = dspy.OutputField()    


class LongFormContent(dspy.Module):
    def __init__(self):
        self.prompt_to_outline = dspy.ChainOfThought(Question2ReportOutline)
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

class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()

class GenerateCitedParagraph(dspy.Signature):
    """Generate a paragraph with citations."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    paragraph = dspy.OutputField(desc="includes citations")

class LongFormQA(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        pred = self.generate_cited_paragraph(context=context, question=question)
        pred = dspy.Prediction(context=context, paragraph=pred.paragraph)
        return pred
    
class LongFormQAWithAssertions(dspy.Module):
    def __init__(self, passages_per_hop=3, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_cited_paragraph = dspy.ChainOfThought(GenerateCitedParagraph)
        self.max_hops = max_hops
    
    def forward(self, question):
        context = []
        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)
        pred = self.generate_cited_paragraph(context=context, question=question)
        pred = dspy.Prediction(context=context, paragraph=pred.paragraph)
        dspy.Suggest(citations_check(pred.paragraph), f"Make sure every 1-2 sentences has citations. If any 1-2 sentences lack citations, add them in 'text... [x].' format.", target_module=GenerateCitedParagraph)
        _, unfaithful_outputs = citation_faithfulness(None, pred, None)
        if unfaithful_outputs:
            unfaithful_pairs = [(output['text'], output['context']) for output in unfaithful_outputs]
            for _, context in unfaithful_pairs:
                dspy.Suggest(len(unfaithful_pairs) == 0, f"Make sure your output is based on the following context: '{context}'.", target_module=GenerateCitedParagraph)
        else:
            return pred
        return pred
    
# ## Example USAGE 
# longformqa_with_assertions = assert_transform_module(LongFormQAWithAssertions().map_named_predictors(Retry), backtrack_handler) 
# evaluate(longformqa_with_assertions)