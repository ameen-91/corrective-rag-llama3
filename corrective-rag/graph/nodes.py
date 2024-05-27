from .chains import rag_chain, retrieval_grader, question_rewriter
from .tools import web_search_tool
from langchain_core.documents import Document
from utils.embeddings import retriever


def retrieve(state):

    question = state["question"]
    documents = retriever.invoke(question)
    return {"question": question, "documents": documents}


def generate(state):

    question = state["question"]
    documents = state["documents"]

    generation = rag_chain.invoke({"question": question, "context": documents})

    return {"question": question, "documents": documents, "generation": generation}


def grade_documents(state):

    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = score["score"]

        if grade == "yes":
            print("---GRADING: DOCUMENT RELEVENT")
            filtered_docs.append(doc)
        else:
            print("---GRADING: DOCUMENT NOT RELEVENT")
            web_search = "Yes"
    return {"question": question, "documents": filtered_docs, "web_search": web_search}


def transform_query(state):

    print("---TRANSFORMING QUERY---")
    question = state["question"]
    documents = state["documents"]

    question = question_rewriter.invoke({"question": question})
    return {"question": question, "documents": documents}


def web_search_node(state):

    print("---PERFORMING WEB SEARCH---")

    question = state["question"]
    documents = state["documents"]

    search_results = web_search_tool.invoke({"query": question})
    filtered_search_results = "\n".join([d["content"] for d in search_results])
    filtered_search_results = Document(page_content=filtered_search_results)

    documents.append(filtered_search_results)

    return {"question": question, "documents": documents}


def decide_to_generate(state):

    print("---DECIDING TO GENERATE---")
    web_search = state["web_search"]

    if web_search == "Yes":
        print("---WEB SEARCH REQUIRED---")
        return "web_search_node"
    else:
        print("---WEB SEARCH NOT REQUIRED---")
        return "generate"
