import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def matching_term_highlight(query, answer, context):
    """
    Highlights the (non-stopword) matching terms in the query (yellow), answer (blue), and context (corresponding color, green for both).

    Args:
        query (str): The query string.
        answer (str): The answer string.
        context (str): The context string.

    Returns:
        tuple: A tuple containing the updated query, answer, and context strings with highlighted keywords.
    """

    stop_words = set(stopwords.words('english'))
    # first go through the query and highlight every term yellow that matches query-context
    for term in query.split(" "):
        if term.lower() in context and term.lower() not in stop_words:
            context = context.replace(term, f"<span style='background-color: yellow'>{term}</span>")
            query = query.replace(term, f"<span style='background-color: yellow'>{term}</span>")
    
    # next go through the answer and highlight every term blue that matches answer-context.
    # if the term is already highlighted in the context, hightligt it green instead
    for term in answer.split(" "):
        if term.lower() in context and term.lower() not in stop_words:
            answer = answer.replace(term, f"<span style='background-color: lightblue'>{term}</span>")
            if f"<span style='background-color: yellow'>{term}</span>" in context:
                context = context.replace(term, f"<span style='background-color: lightgreen'>{term}</span>")
            else:
                context = context.replace(term, f"<span style='background-color: lightblue'>{term}</span>")
    return query, answer, context

def format_answer(question, answer, vstore, score_threshold):
    """
    Formats the explained answer with source information and relevant context based on the question and similarity scores.

    Args:
        question (str): The question to search for.
        answer (str): The answer to format.
        vstore: The similarity search object.
        score_threshold (float): The threshold for similarity scores.

    Returns:
        str: The formatted answer with relevant context.
    """
    results = vstore.similarity_search_with_relevance_scores(question)
    docs, scores = zip(*results)
    format_context = ""
    for i, doc in enumerate(docs):
        if scores[i] > score_threshold:
            metadata = doc.metadata
            format_context += f"""
            Context {i+1} (score: {round(scores[i],3)})<br> 
            Source {metadata.get('source')} page {metadata.get('page')}<br>
            Content:<br>{doc.page_content}<br><br>
            """
    lit_question, lit_answer, lit_context = matching_term_highlight(question, answer, format_context)
    lit_context = lit_context.replace("\n\n", "<br><br>" )
    full_response = f"""
        Question: <br>{lit_question}<br><br>
        Answer: <br>{lit_answer}
        <br><br>
        {lit_context}
        
    """
    return full_response