import streamlit as st
import arxiv
from groq import Groq
import os
import requests
import base64

# 1. Setup
st.set_page_config(page_title="Simple Q-Scout", page_icon="🔬", layout="wide")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY is missing.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# 2. Data Acquisition
def get_paper_data(arxiv_id):
    arxiv_client = arxiv.Client(page_size=1, delay_seconds=3, num_retries=5)
    search = arxiv.Search(id_list=[arxiv_id])
    try:
        paper = next(arxiv_client.results(search))
        # Use the official summary to avoid "Request Too Large" errors
        summary_context = f"Title: {paper.title}\n\nAbstract: {paper.summary}"
        
        # Download the PDF bytes to bypass browser blocking
        response = requests.get(paper.pdf_url)
        base64_pdf = base64.b64encode(response.content).decode('utf-8')
        
        return paper.title, summary_context, base64_pdf
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None, None

# 3. State management
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_id' not in st.session_state:
    st.session_state.current_id = None

# Sidebar for controls
with st.sidebar:
    st.header("Search")
    arxiv_id = st.text_input("Enter ArXiv ID", "1706.03762")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# 4. Main Layout
col1, col2 = st.columns([1, 1], gap="large")

if arxiv_id:
    # Load paper data
    if st.session_state.current_id != arxiv_id:
        with st.spinner("Downloading PDF and preparing assistant..."):
            title, summary, pdf_b64 = get_paper_data(arxiv_id)
            if title:
                st.session_state.paper_title = title
                st.session_state.paper_summary = summary
                st.session_state.pdf_ref = pdf_b64
                st.session_state.current_id = arxiv_id
                st.session_state.chat_history = []
            else:
                st.stop()

    # --- LEFT COLUMN: CHAT ---
    with col1:
        st.subheader(f"💬 {st.session_state.paper_title}")
        
        # Scrollable container for chat history
        chat_box = st.container(height=600)
        
        with chat_box:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input - strictly appears at bottom of col1
        if prompt := st.chat_input("Ask about the paper..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            with chat_box:
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    messages = [
                        {"role": "system", "content": f"You are a research assistant. Context:\n{st.session_state.paper_summary}"},
                    ]
                    messages.extend(st.session_state.chat_history[-6:])
                    
                    try:
                        response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=messages,
                        )
                        answer = response.choices[0].message.content
                        st.markdown(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(f"Error: {e}")
            st.rerun()

    # --- RIGHT COLUMN: PDF VIEWER (Fixed Blocking) ---
    with col2:
        st.subheader("📄 Full PDF")
        if 'pdf_ref' in st.session_state:
            # We embed the PDF as data so Edge/Chrome won't block it
            pdf_display = f'<embed src="data:application/pdf;base64,{st.session_state.pdf_ref}" width="100%" height="800px" type="application/pdf">'
            st.markdown(pdf_display, unsafe_allow_html=True)

else:
    st.info("Enter an ArXiv ID in the sidebar to begin.")