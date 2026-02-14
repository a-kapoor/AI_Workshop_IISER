import streamlit as st
import arxiv
import fitz  # PyMuPDF
from groq import Groq, RateLimitError
from streamlit_mic_recorder import mic_recorder
from kokoro import KPipeline
from tavily import TavilyClient
import soundfile as sf
import os
import requests
import io
import re
import time
import numpy as np
from datetime import datetime, timedelta, timezone

# --- CONFIG & THEME ---
st.set_page_config(page_title="Q-Scout Pro", page_icon="🔬", layout="wide")

# API Keys (Ensure these are set in your environment or secrets)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY is missing. Please set it in your environment variables.")
    st.stop()

client_groq = Groq(api_key=GROQ_API_KEY)

# --- STATE MANAGEMENT ---
# Research Mode State
if "research_data" not in st.session_state:
    st.session_state.research_data = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Single Paper Chat Mode State
if "single_paper_data" not in st.session_state:
    st.session_state.single_paper_data = None
if "single_paper_chat_history" not in st.session_state:
    st.session_state.single_paper_chat_history = []

# --- CACHED RESOURCES ---
@st.cache_resource
def load_tts_pipeline():
    try:
        return KPipeline(lang_code='a')
    except Exception as e:
        # Silently fail or warn if TTS isn't essential
        print(f"TTS Model load failed: {e}")
        return None

@st.cache_resource
def get_arxiv_client():
    return arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=3)

# --- HELPER FUNCTIONS ---

def clean_markdown_for_speech(text):
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+\s', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    text = re.sub(r'`+', '', text)
    text = re.sub(r'^\s*[-*]\s+', '', text, flags=re.MULTILINE)
    return text.strip()

def safe_groq_call(messages, model_choice="llama-3.3-70b-versatile"):
    try:
        return client_groq.chat.completions.create(model=model_choice, messages=messages)
    except RateLimitError:
        time.sleep(1) 
        return client_groq.chat.completions.create(model="llama-3.1-8b-instant", messages=messages)
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def generate_audio_stream(text):
    try:
        speakable_text = clean_markdown_for_speech(text)
        pipeline = load_tts_pipeline()
        if not pipeline: return None
        generator = pipeline(speakable_text, voice='af_bella', speed=1.1)
        full_audio = [audio for _, _, audio in generator]
        if full_audio:
            combined_audio = np.concatenate(full_audio)
            buffer = io.BytesIO()
            sf.write(buffer, combined_audio, 24000, format='WAV')
            buffer.seek(0)
            return buffer
    except Exception as e:
        st.error(f"Audio Engine Error: {e}")
    return None

def smart_pdf_extract(pdf_url, full_extract=False):
    """
    Downloads PDF. 
    full_extract=False -> Extracts first 3 + last 2 pages (Research Mode).
    full_extract=True -> Extracts ALL pages (Single Paper Mode).
    """
    try:
        headers = {'User-Agent': 'Q-Scout-Research/4.0'}
        response = requests.get(pdf_url, timeout=10, headers=headers)
        content = response.content
        extracted_text = ""
        
        with fitz.open(stream=io.BytesIO(content), filetype="pdf") as doc:
            total_pages = len(doc)
            
            if full_extract:
                # Read all pages for single paper deep dive
                extracted_text = "\n".join([page.get_text() for page in doc])
            else:
                # Optimized reading for multi-paper research
                pages_to_read = [0, 1, 2]
                if total_pages > 5:
                    pages_to_read.extend([total_pages-2, total_pages-1])
                pages_to_read = sorted(list(set([p for p in pages_to_read if p < total_pages])))
                extracted_text = "\n".join([doc[p].get_text() for p in pages_to_read])
            
        return content, extracted_text
    except Exception as e:
        print(f"PDF Extract Error: {e}")
        return None, ""

def web_search_tavily(query, max_results=3):
    if not TAVILY_API_KEY: return "⚠️ TAVILY_API_KEY missing."
    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        response = tavily_client.search(query=query, search_depth="advanced", max_results=max_results)
        context_str = ""
        for result in response.get("results", []):
            context_str += f"Source: {result['url']}\nTitle: {result['title']}\nContent: {result['content']}\n\n"
        return context_str
    except Exception as e:
        return f"Tavily Search Error: {e}"

# --- UI LAYER ---

with st.sidebar:
    st.header("🔬 Q-Scout Pro")
    
    # APP MODE SELECTION
    app_mode = st.selectbox("Select Mode", ["🚀 Topic Research", "📄 Chat with Paper"])
    
    st.divider()

    if app_mode == "🚀 Topic Research":
        st.subheader("Topic Controls")
        input_mode = st.radio("Input Method", ["🎤 Voice", "⌨️ Text"], horizontal=True)
        
        start_research = False
        query_text = ""

        if input_mode == "🎤 Voice":
            audio_input = mic_recorder(start_prompt="🔴 Record Query", stop_prompt="⏹️ Process", key='vox_recorder')
            if audio_input:
                audio_file = io.BytesIO(audio_input['bytes'])
                audio_file.name = "audio.wav"
                transcription = client_groq.audio.transcriptions.create(file=audio_file, model="whisper-large-v3", response_format="text")
                query_text = transcription
                start_research = True
        else:
            user_text = st.text_area("Research Query", placeholder="e.g. 'Sparse attention transformers'")
            if st.button("🚀 Run Research"):
                query_text = user_text
                start_research = True

        with st.expander("⚙️ Search Filters"):
            lookback = st.slider("ArXiv Recency (Days)", 7, 365, 30)
            max_docs = st.number_input("Max Papers", 1, 10, 3)
            model_select = st.selectbox("Reasoning Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"], key="model_topic")

    elif app_mode == "📄 Chat with Paper":
        st.subheader("Paper Controls")
        arxiv_id_input = st.text_input("ArXiv ID", placeholder="e.g. 1706.03762")
        load_paper_btn = st.button("📥 Load Paper")
        model_select_paper = st.selectbox("Reasoning Model", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"], key="model_paper")


# --- MAIN LOGIC ---

# ==========================================
# MODE 1: TOPIC RESEARCH
# ==========================================
if app_mode == "🚀 Topic Research":
    st.title("🚀 Topic Deep Dive")

    # Retrieve currently loaded topic to prevent loop
    current_topic = st.session_state.research_data.get("topic") if st.session_state.research_data else ""

    # 1. TRIGGER RESEARCH
    # We add a check: Is 'start_research' True AND is this a NEW topic?
    if start_research and query_text and query_text != current_topic:
        st.session_state.research_data = None
        st.session_state.chat_history = []
        
        st.markdown(f"### 🔎 Investigating: *{query_text}*")
        
        with st.status("Running Research Agent...", expanded=True) as status:
            # A. TAVILY
            status.write("🌐 Tavily is scanning the web...")
            web_data = web_search_tavily(query_text)
            
            # B. ARXIV
            status.write("📄 Querying ArXiv database...")
            now = datetime.now(timezone.utc)
            start_date = (now - timedelta(days=lookback)).strftime("%Y%m%d0000")
            end_date = now.strftime("%Y%m%d2359")
            search_query = f'all:"{query_text}" AND submittedDate:[{start_date} TO {end_date}]'
            
            client_arxiv = get_arxiv_client()
            search = arxiv.Search(query=search_query, max_results=max_docs, sort_by=arxiv.SortCriterion.Relevance)
            papers_found = list(client_arxiv.results(search))
            
            # C. PROCESS
            intel_pool = [f"--- WEB CONTEXT (TAVILY) ---\n{web_data}"]
            sources = []
            
            if not papers_found:
                status.write("⚠️ No recent ArXiv papers found. Relying on web search.")
            
            for paper in papers_found:
                status.write(f"📥 Reading: {paper.title[:40]}...")
                # Partial extract for topic research
                _, full_text = smart_pdf_extract(paper.pdf_url, full_extract=False)
                
                if full_text:
                    sources.append({
                        "title": paper.title,
                        "url": paper.pdf_url,
                        "authors": ", ".join([a.name for a in paper.authors]),
                        "year": paper.published.year,
                        "summary": paper.summary
                    })
                    paper_prompt = f"Analyze text from '{paper.title}':\n{full_text[:12000]}\nTask: Extract 1. Methodology 2. Results 3. Novelty"
                    res = safe_groq_call([{"role": "user", "content": paper_prompt}], model_choice="llama-3.1-8b-instant")
                    if res:
                        intel_pool.append(f"--- PAPER: {paper.title} ---\n{res.choices[0].message.content}")

            # D. SYNTHESIS
            status.write("🧠 Synthesizing final report...")
            research_context = "\n\n".join(intel_pool)
            final_prompt = f"""
            You are a Principal Researcher. User Query: "{query_text}"
            Data Sources: {research_context}
            Task: Write a high-density executive briefing (300 words). Focus on NOVELTY & GAPS.
            """
            final_res = safe_groq_call([{"role": "user", "content": final_prompt}], model_choice=model_select)
            script = final_res.choices[0].message.content if final_res else "Analysis failed."
            
            st.session_state.research_data = {
                "topic": query_text,
                "script": script,
                "context": research_context,
                "sources": sources
            }
            status.update(label="✅ Research Complete", state="complete", expanded=False)
            st.rerun()

    # 2. RENDER TOPIC RESULTS
    if st.session_state.research_data:
        data = st.session_state.research_data
        st.markdown(f"### 📂 Results for: *{data['topic']}*")

        tab_brief, tab_sources, tab_chat = st.tabs(["🎙️ Briefing", "📚 Sources", "💬 Q&A"])

        with tab_brief:
            st.markdown(data['script'])
            st.divider()
            if st.button("🔊 Generate Audio"):
                with st.spinner("Synthesizing..."):
                    audio_stream = generate_audio_stream(data['script'])
                    if audio_stream: st.audio(audio_stream, format="audio/wav", autoplay=True)

        with tab_sources:
            for s in data['sources']:
                with st.expander(f"📄 {s['title']}"):
                    st.write(f"**Authors:** {s['authors']}")
                    st.markdown(f"[Open PDF]({s['url']})")

        with tab_chat:
            st.caption("Chat with these results.")
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]): st.write(msg["content"])

            if prompt := st.chat_input("Ask about this topic..."):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"): st.write(prompt)
                
                with st.chat_message("assistant"):
                    rag_messages = [
                        {"role": "system", "content": f"Answer using ONLY context:\n{data['context']}"}
                    ] + st.session_state.chat_history[-4:]
                    
                    response = safe_groq_call(rag_messages, model_choice=model_select)
                    if response:
                        reply = response.choices[0].message.content
                        st.write(reply)
                        st.session_state.chat_history.append({"role": "assistant", "content": reply})
    else:
        st.info("👈 Start a research session from the sidebar.")


# ==========================================
# MODE 2: CHAT WITH PAPER
# ==========================================
elif app_mode == "📄 Chat with Paper":
    st.title("📄 Chat with Paper")

    # 1. LOAD PAPER LOGIC
    # Similar check here: Only run if we haven't already loaded this exact paper
    current_paper_title = st.session_state.single_paper_data['title'] if st.session_state.single_paper_data else ""
    
    if load_paper_btn and arxiv_id_input:
        
        st.session_state.single_paper_data = None
        st.session_state.single_paper_chat_history = []
        
        with st.status("Fetching Paper...", expanded=True) as status:
            try:
                # Resolve ID
                client_arxiv = get_arxiv_client()
                search = arxiv.Search(id_list=[arxiv_id_input])
                paper = next(client_arxiv.results(search))
                
                status.write(f"📥 Downloading: {paper.title}...")
                
                # Full Extraction for single paper
                _, full_text = smart_pdf_extract(paper.pdf_url, full_extract=True)
                
                if full_text:
                    st.session_state.single_paper_data = {
                        "title": paper.title,
                        "url": paper.pdf_url,
                        "authors": ", ".join([a.name for a in paper.authors]),
                        "summary": paper.summary,
                        "full_text": full_text
                    }
                    status.update(label="✅ Paper Loaded", state="complete", expanded=False)
                    st.rerun()
                else:
                    status.update(label="❌ Failed to extract text", state="error")
            except Exception as e:
                st.error(f"Error loading paper: {e}")

    # 2. RENDER PAPER INTERFACE
    if st.session_state.single_paper_data:
        pdata = st.session_state.single_paper_data
        
        st.markdown(f"## 📄 {pdata['title']}")
        st.caption(f"**Authors:** {pdata['authors']} | [Open PDF]({pdata['url']})")
        
        with st.expander("📝 Abstract", expanded=False):
            st.write(pdata['summary'])
            
        st.divider()
        
        # Chat Interface
        for msg in st.session_state.single_paper_chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
        
        if prompt := st.chat_input("Ask a question about this paper..."):
            st.session_state.single_paper_chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing text..."):
                    context_text = pdata['full_text'][:100000] 
                    
                    sys_prompt = (
                        f"You are an expert academic assistant. "
                        f"Answer the user's question solely based on the paper: '{pdata['title']}'. "
                        f"If the answer is not in the text, state that explicitly.\n\n"
                        f"PAPER TEXT:\n{context_text}"
                    )
                    
                    messages = [{"role": "system", "content": sys_prompt}]
                    messages.extend(st.session_state.single_paper_chat_history[-5:])
                    
                    response = safe_groq_call(messages, model_choice=model_select_paper)
                    
                    if response:
                        reply = response.choices[0].message.content
                        st.write(reply)
                        st.session_state.single_paper_chat_history.append({"role": "assistant", "content": reply})

    else:
        st.info("👈 Enter an ArXiv ID (e.g., `1706.03762`) in the sidebar to begin.")