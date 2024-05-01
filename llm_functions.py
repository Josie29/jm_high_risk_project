from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from typing import Tuple, List

def format_docs(docs):
    """
    Formats a list of documents.

    Args:
        docs (list): A list of documents.

    Returns:
        str: A formatted string containing the page content of each document, separated by two newlines.
    """ 
    return "\n\n".join(doc.page_content for doc in docs)


def perform_rag(query: str, llm, retriever) -> Tuple[str, List]:
    """

    Performs the RAG (Retrieval-Augmented Generation) process to generate an answer and context for a given query.

    Args:
        query (str): The query to be answered.
        llm: The language model used for generation.
        retriever: The retriever used for context retrieval.

    Returns:
        Tuple[str, str]: A tuple containing the generated answer and the retrieved context.
    """

    prompt_template = """
    Answer the question based only on the supplied context. If you don't know the answer from just the context, say 'the answer is not provided in the context'.
    Context: {context}
    Question: {question}
    Your answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain_from_docs = (
        RunnablePassthrough.assign(context = (lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer = chain_from_docs)

    response = chain_with_source.invoke(query)
    answer = response.get('answer')
    context_doc_list = response.get('context')
    context = format_docs(context_doc_list)
    return answer, context