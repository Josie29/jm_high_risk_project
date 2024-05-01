from typing import List

from llm_functions import perform_rag

import pandas as pd
from langchain_core.documents import Document
from ragas.testset.generator import TestsetGenerator, TestDataset
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas.metrics import context_recall, context_precision, answer_correctness
from ragas import evaluate
from datasets import Dataset

def make_test_set(docs: List[Document], test_size: int, llm, embedding) -> TestDataset:
    """
    Generate a test dataset using the given documents, test size, language model, and embedding.

    Args:
        docs (List[Document]): The list of documents to generate the test dataset from.
        test_size (int): The size of the test dataset to generate.
        llm: The language model used for generating the test dataset.
        embedding: The embedding used for generating the test dataset.

    Returns:
        TestDataset: The generated test dataset.
    """
    generator = TestsetGenerator.from_langchain(
        generator_llm=llm,
        critic_llm=llm,
        embeddings=embedding,
    )
    distribution_dict = {simple: 1.0, reasoning: 0.0, multi_context: 0.0}
    testset = generator.generate_with_langchain_docs(documents=docs, test_size=test_size, distributions=distribution_dict)
    return testset

def add_llm_answers_to_dataset(testset: TestDataset, retriever, llm):
    """
    Adds LLM answers to the given test dataset.

    Args:
        testset (TestDataset): The test dataset to add LLM answers to.
        retriever: The retriever object used for retrieving relevant contexts.
        llm: The language model used for generating answers.

    Returns:
        Dataset: The updated dataset with LLM answers added.
    """
    test_df = testset.to_pandas()
    test_df[['answer', 'context']] = test_df["question"].apply(lambda q: pd.Series(perform_rag(q, llm, retriever)))
    dataset = Dataset.from_pandas(test_df)
    return dataset

def evaluate_dataset(dataset: TestDataset, output_path: str = None):
    """
    Evaluates the given dataset and returns results with the option for outputting to excel.

    Args:
        dataset (TestDataset): The dataset to be evaluated.
        output_path (str, optional): The path to save the evaluation results as an Excel file. Defaults to None.

    Returns:
        tuple: A tuple containing the evaluation result object and the evaluation results as a pandas DataFrame.
    """
    result = evaluate(dataset, 
                      metrics=[
                                context_precision,  
                                context_recall,
                                answer_correctness
                              ],
                    )
    if output_path:
        results_df = result.to_pandas()
        results_df.to_excel(output_path)
    return result