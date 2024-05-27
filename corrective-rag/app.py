import streamlit as st
from graph.graph import corrective_rag, simple_rag
from utils.embeddings import load_doc


st.title("Corrective Rag App")

simple_rag_toggle = st.toggle("Simple RAG", False)

if simple_rag_toggle:
    pipeline = simple_rag()
else:
    pipeline = corrective_rag()


url = st.text_input("Enter the URL of the document:")
file = st.file_uploader(
    "Upload a document",
    type=["pdf", "docx"],
    accept_multiple_files=True,
)
if file:
    retriever = load_doc(file, type="file")
    if retriever:
        st.success("Document uploaded successfully!")
elif url:
    retriever = load_doc(url, type="url")
    if retriever:
        st.success("URL uploaded successfully!")


question = st.chat_input("Enter your question:")

if question:
    with st.chat_message(name="User"):
        st.write(question)
    state = {"question": question}
    with st.status("Generating Response...", expanded=True) as status:
        for output in pipeline.stream(state):
            for key, value in output.items():
                status.text(f"Processing {key}...")
    status.update(label="Response Generated!", state="complete", expanded=False)
    with st.chat_message(name="AI"):
        st.write(value["generation"])
