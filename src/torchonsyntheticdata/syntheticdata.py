from typing import Optional
from typing import List
import random
from pydantic import BaseModel
import dspy
# from dspy.signatures import (
#     ExplainTask,
#     GenerateFieldDescription,
#     GenerateInputFieldsData,
#     GenerateOutputFieldsData,
#     GetFeedbackOnGeneration,
#     UnderstandTask,
#     UpdateTaskDescriptionBasedOnFeedback,
# # )
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
class DescriptionSignature(dspy.Signature):
    """Write a simple search query that will require a complex answer.
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
    
class SyntheticDataHandler:
    def __init__(self, examples: Optional[List[dspy.Example]] = None):
        self.generator = SyntheticDataGenerator(examples=examples)

    def generate_data(self, sample_size: int):
        return self.generator.generate(sample_size=sample_size)

    def configure_dspy_settings(lm_model,retriever):
        dspy.settings.configure(rm=retriever, lm=lm_model)

    def prepare_datasets(dataset):
        trainset = [x.with_inputs('question') for x in dataset.train]
        devset = [x.with_inputs('question') for x in dataset.dev]
        testset = [x.with_inputs('question') for x in dataset.test]
        return trainset, devset, testset
