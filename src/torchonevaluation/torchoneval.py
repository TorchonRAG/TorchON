# ./src/torchonevaluation/torchonevaluation.py
import dspy
class CheckCitationFaithfulness(dspy.Signature):
    """Verify that the text is based on the provided context."""
    context = dspy.InputField(desc="may contain relevant facts")
    text = dspy.InputField(desc="between 1 to 2 sentences")
    faithfulness = dspy.OutputField(desc="boolean indicating if text is faithful to context")
class TorchonEval:
    def extract_cited_titles_from_paragraph(paragraph, context):
        cited_indices = [int(m.group(1)) for m in re.finditer(r'\[(\d+)\]\.', paragraph)]
        cited_indices = [index - 1 for index in cited_indices if index <= len(context)]
        cited_titles = [context[index].split(' | ')[0] for index in cited_indices]
        return cited_titles

    def calculate_recall(example, pred, trace=None):
        gold_titles = set(example['gold_titles'])
        found_cited_titles = set(extract_cited_titles_from_paragraph(pred.paragraph, pred.context))
        intersection = gold_titles.intersection(found_cited_titles)
        recall = len(intersection) / len(gold_titles) if gold_titles else 0
        return recall

    def calculate_precision(example, pred, trace=None):
        gold_titles = set(example['gold_titles'])
        found_cited_titles = set(extract_cited_titles_from_paragraph(pred.paragraph, pred.context))
        intersection = gold_titles.intersection(found_cited_titles)
        precision = len(intersection) / len(found_cited_titles) if found_cited_titles else 0
        return precision

    def answer_correctness(example, pred, trace=None):
        assert hasattr(example, 'answer'), "Example does not have 'answer'."
        normalized_context = normalize_text(pred.paragraph)
        if isinstance(example.answer, str):
            gold_answers = [example.answer]
        elif isinstance(example.answer, list):
            gold_answers = example.answer
        else:
            raise ValueError("'example.answer' is not string or list.")
        return 1 if any(normalize_text(answer) in normalized_context for answer in gold_answers) else 0   
    def citation_faithfulness(example, pred, trace):
        paragraph, context = pred.paragraph, pred.context
        citation_dict = extract_text_by_citation(paragraph)
        if not citation_dict:
            return False, None
        context_dict = {str(i): context[i].split(' | ')[1] for i in range(len(context))}
        faithfulness_results = []
        unfaithful_citations = []
        check_citation_faithfulness = dspy.ChainOfThought(CheckCitationFaithfulness)
        for citation_num, texts in citation_dict.items():
            if citation_num not in context_dict:
                continue
            current_context = context_dict[citation_num]
            for text in texts:
                try:
                    result = check_citation_faithfulness(context=current_context, text=text)
                    is_faithful = result.faithfulness.lower() == 'true'
                    faithfulness_results.append(is_faithful)
                    if not is_faithful:
                        unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'context': current_context})
                except ValueError as e:
                    faithfulness_results.append(False)
                    unfaithful_citations.append({'paragraph': paragraph, 'text': text, 'error': str(e)})
        final_faithfulness = all(faithfulness_results)
        if not faithfulness_results:
            return False, None
        return final_faithfulness, unfaithful_citations
    def evaluate(module):
        correctness_values = []
        recall_values = []
        precision_values = []
        citation_faithfulness_values = []
        for i in range(len(devset)):
            example = devset[i]
            try:
                pred = module(question=example.question)
                correctness_values.append(answer_correctness(example, pred))            
                citation_faithfulness_score, _ = citation_faithfulness(None, pred, None)
                citation_faithfulness_values.append(citation_faithfulness_score)
                recall = calculate_recall(example, pred)
                precision = calculate_precision(example, pred)
                recall_values.append(recall)
                precision_values.append(precision)
            except Exception as e:
                print(f"Failed generation with error: {e}")

        average_correctness = sum(correctness_values) / len(devset) if correctness_values else 0
        average_recall = sum(recall_values) / len(devset) if recall_values else 0
        average_precision = sum(precision_values) / len(devset) if precision_values else 0
        average_citation_faithfulness = sum(citation_faithfulness_values) / len(devset) if citation_faithfulness_values else 0

        print(f"Average Correctness: {average_correctness}")
        print(f"Average Recall: {average_recall}")
        print(f"Average Precision: {average_precision}")
        print(f"Average Citation Faithfulness: {average_citation_faithfulness}")

# ## EXAMPLE USAGE
#         longformqa = LongFormQA()
#         evaluate(longformqa)