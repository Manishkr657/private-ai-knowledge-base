
import requests
import streamlit as st

BASE_URL = "http://127.0.0.1:8000"
ASK_URL = f"{BASE_URL}/ask"
DOCUMENTS_URL = f"{BASE_URL}/documents"
UPLOAD_URL = f"{DOCUMENTS_URL}/upload"

st.set_page_config(
    page_title="Personal AI Assistant",
    page_icon="🧠",
    layout="centered",
)

st.title("Personal AI Assistant")
st.caption("Ask questions about your own knowledge base.")

# Document management
with st.sidebar:
    st.header("Knowledge Base")
    st.caption("Upload your own TXT documents.")

    uploaded_file = st.file_uploader(
        "Choose a text file",
        type=["txt", "pdf"],
    )

    if st.button("Upload and index", disabled=uploaded_file is None):
        try:
            with st.spinner("Indexing document..."):
                response = requests.post(
                    UPLOAD_URL,
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "text/plain",
                        )
                    },
                    timeout=180,
                )
                response.raise_for_status()

            result = response.json()
            st.success(
                f"Indexed {result['filename']} "
                f"({result['chunks']} chunk(s))."
            )
            st.rerun()

        except requests.RequestException as exc:
            st.error("Upload failed. Check that FastAPI is running.")
            st.caption(str(exc))

    st.divider()
    st.subheader("Indexed documents")

    if st.button("Refresh document list"):
        st.rerun()

    try:
        response = requests.get(DOCUMENTS_URL, timeout=10)
        response.raise_for_status()
        documents = response.json().get("documents", [])

        
        
        if documents:
            for document in documents:
                source_id = document["source_id"]
                filename = document["filename"]

                col_name, col_remove = st.columns([5, 1])

                with col_name:
                    st.write(f"**{filename}**")
                    st.caption(f"{document['chunks']} text chunk(s)")

                with col_remove:
                    if st.button(
                        "×",
                        key=f"remove_{source_id}",
                        help=f"Remove {filename} from the knowledge base",
                    ):
                        try:
                            response = requests.delete(
                                f"{DOCUMENTS_URL}/{source_id}",
                                timeout=15,
                            )
                            response.raise_for_status()
                            st.rerun()

                        except requests.RequestException:
                            st.error(f"Could not remove {filename}.")
        else:
            st.info("No documents indexed yet.")



    except requests.RequestException:
        st.warning("Cannot load documents. Is FastAPI running?")

# Chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your knowledge base..."):
            try:
                response = requests.post(
                    ASK_URL,
                    json={"question": question},
                    timeout=180,
                )
                response.raise_for_status()
                data = response.json()

                answer = data["answer"]
                st.markdown(answer)

                
                
                sources = data.get("sources", [])
                if sources:
                    with st.expander("Sources used to answer"):
                        st.caption(
                            "These are the passages retrieved "
                            "from your knowledge base."
                        )

                        for index, source in enumerate(sources, start=1):
                            filename = source.get("filename", "Unknown document")
                            passage = source.get("text", "")

                            st.markdown(f"**Source {index}: {filename}**")
                            st.write(passage)



                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except requests.RequestException as exc:
                st.error(
                    "Could not reach the AI backend. "
                    "Check that FastAPI is running at "
                    f"{BASE_URL}."
                )
                st.caption(str(exc))
