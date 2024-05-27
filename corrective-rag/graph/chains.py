from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate


def create_main_chain():

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOllama(model="llama3:instruct", temperature=0)
    rag_chain = prompt | llm | StrOutputParser()
    return rag_chain


def create_retrieval_grader_chain():

    grading_llm = ChatOllama(model="llama3:instruct", format="json", temperature=0)
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
        input_variables=["question", "document"],
    )
    retrieval_grader = prompt | grading_llm | JsonOutputParser()
    return retrieval_grader


def create_question_rewriter_chain():

    llm = ChatOllama(model="llama3:instruct", temperature=0)
    re_write_prompt = PromptTemplate(
        template="""You a question re-writer that converts an input question to a better version that is optimized \n 
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}. Give only the improved question with no preamble or explanation: \n """,
        input_variables=["generation", "question"],
    )
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    return question_rewriter


rag_chain = create_main_chain()
retrieval_grader = create_retrieval_grader_chain()
question_rewriter = create_question_rewriter_chain()
