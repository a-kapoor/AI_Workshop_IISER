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
        st.error(f"Error fetching {arxiv_id}: {e}")
        return None, None, None

# 3. State management
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_ids' not in st.session_state:
    st.session_state.current_ids = []
if 'papers_data' not in st.session_state:
    st.session_state.papers_data = []

# Sidebar for controls
with st.sidebar:
    st.header("Search")
    # Accept multiple IDs separated by commas
    arxiv_ids_input = st.text_input("Enter up to 3 ArXiv IDs (comma-separated)", "")
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Parse the input IDs and limit to 3
raw_ids = [x.strip() for x in arxiv_ids_input.split(",") if x.strip()]
input_ids = raw_ids[:3]

# 4. Main Layout
col1, col2 = st.columns([1, 1], gap="large")

if input_ids:
    # Load paper data if the list of IDs has changed
    if st.session_state.current_ids != input_ids:
        with st.spinner(f"Downloading {len(input_ids)} PDF(s) and preparing assistant..."):
            st.session_state.papers_data = []
            
            for aid in input_ids:
                title, summary, pdf_b64 = get_paper_data(aid)
                if title:
                    st.session_state.papers_data.append({
                        "id": aid,
                        "title": title,
                        "summary": summary,
                        "pdf_ref": pdf_b64
                    })
            
            st.session_state.current_ids = input_ids
            st.session_state.chat_history = []

    # Only proceed if we successfully loaded at least one paper
    if st.session_state.papers_data:
        # Combine summaries for the LLM context
        combined_context = "\n\n---\n\n".join(
            [f"Paper ID {p['id']}:\n{p['summary']}" for p in st.session_state.papers_data]
        )

        # --- LEFT COLUMN: CHAT ---
        with col1:
            st.subheader(f"💬 Chatting with {len(st.session_state.papers_data)} Paper(s)")
            st.caption(" | ".join([p["title"][:50] + "..." for p in st.session_state.papers_data]))
            
            # Scrollable container for chat history
            chat_box = st.container(height=600)
            
            with chat_box:
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

            # Chat input - strictly appears at bottom of col1
            if prompt := st.chat_input("Ask about the papers..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                with chat_box:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        messages = [
                            {"role": "system", "content": f"You are a research assistant. Context:\n{combined_context}"},
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

        # --- RIGHT COLUMN: PDF VIEWER (Selectbox instead of Tabs) ---
        with col2:
            st.subheader("📄 Full PDFs")
            
            # Create a dictionary mapping a readable name to the paper data
            paper_options = {
                f"{p['id']} - {p['title'][:40]}...": p 
                for p in st.session_state.papers_data
            }
            
            # Let the user choose which PDF to view
            selected_paper_key = st.selectbox(
                "Select a paper to read:", 
                options=list(paper_options.keys())
            )
            
            if selected_paper_key:
                active_paper = paper_options[selected_paper_key]
                pdf_ref = active_paper["pdf_ref"]
                
                # Because only one renders at a time, <embed> works perfectly here
                pdf_display = f'''
                    <embed 
                        src="data:application/pdf;base64,{pdf_ref}" 
                        width="100%" 
                        height="800px" 
                        type="application/pdf"
                    >
                '''
                st.markdown(pdf_display, unsafe_allow_html=True)

else:
    st.info("Enter ArXiv IDs in the sidebar to begin.")