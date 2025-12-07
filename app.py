import streamlit as st
import PyPDF2
import os
import time
import random
import contextlib
import json
import base64
import io
import re
from datetime import datetime, timedelta
from PIL import Image

# --- Try-Except Imports for robustness ---
try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, NotFound, InvalidArgument
except ImportError:
    pass

try:
    from openai import OpenAI, OpenAIError
except ImportError:
    pass

try:
    import anthropic
except ImportError:
    pass

try:
    import docx
except ImportError:
    pass

# ==============================================================================
# 🔑 API KEY CONFIGURATION (AUTO-DETECT)
# ==============================================================================
try:
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    API_KEY = "" # Enter your key here for local testing
# ==============================================================================

# --- Configuration & Setup ---
st.set_page_config(page_title="AI Tech Interviewer Pro", layout="wide")

# --- Custom CSS for Night Blue Theme ---
st.markdown("""
    <style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #1e293b; }
    h1, h2, h3, h4, h5, h6 { color: #f8fafc !important; }
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #334155; color: #f8fafc; border-color: #475569;
    }
    .stSelectbox > div > div > div { background-color: #334155; color: #f8fafc; }
    .stButton > button { background-color: #3b82f6; color: white; border: none; border-radius: 6px; font-weight: bold; }
    .stButton > button:hover { background-color: #2563eb; }
    .timer-display {
        position: fixed; bottom: 20px; right: 20px; background-color: #dc2626;
        color: white; padding: 10px 20px; border-radius: 8px; font-weight: bold;
        z-index: 9999; box-shadow: 0 4px 6px rgba(0,0,0,0.3); font-family: monospace; font-size: 18px;
    }
    .processing-overlay {
        position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
        background: rgba(15, 23, 42, 0.95); z-index: 999999;
        display: flex; flex-direction: column; justify-content: center; align-items: center; color: white;
    }
    .stRadio label { color: #e2e8f0 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- Universal LLM Caller (Robust with Retry) ---
def call_llm(provider, model_name, api_key, prompt, image_data=None):
    if not api_key: return "Error: API Key missing."
    
    # Force Google Gemini for stability in this demo
    if provider == "Google Gemini":
        try:
            genai.configure(api_key=api_key)
            # Expanded fallback list with delay logic
            candidates = [
                'gemini-2.0-flash-exp', 
                'gemini-1.5-flash', 
                'gemini-1.5-pro',
                'gemini-2.5-pro',
                'gemini-2.5-flash'
            ]
            
            last_error = ""
            for m in candidates:
                # Retry each model up to 2 times
                for attempt in range(2):
                    try:
                        model = genai.GenerativeModel(m)
                        if image_data:
                            response = model.generate_content([prompt, image_data])
                        else:
                            response = model.generate_content(prompt)
                        return response.text
                    except Exception as e:
                        last_error = str(e)
                        # If Rate Limit (429), wait longer
                        if "429" in str(e) or "ResourceExhausted" in str(e):
                            time.sleep(2)
                        else:
                            time.sleep(1)
                        continue
            return f"Error: System busy. Please wait 10s and try again. (Details: {last_error})"
        except Exception as e:
            return f"Gemini Connection Error: {str(e)}"
    return "Error: Provider not supported in this version."

# --- Helper Functions ---

def extract_text_from_docx(file):
    try:
        import docx
        doc = docx.Document(file)
        return '\n'.join([p.text for p in doc.paragraphs])
    except: return "Error reading .docx"

def process_uploaded_file(uploaded_file, provider, user_api_key):
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        elif "word" in uploaded_file.type:
            return extract_text_from_docx(uploaded_file)
        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            image = Image.open(uploaded_file)
            prompt = "Transcribe this resume text exactly. English only."
            return call_llm(provider, "gemini-2.0-flash-exp", user_api_key, prompt, image_data=image)
    except Exception as e: return f"Error: {e}"
    return None

def check_cv_elements(text):
    # FORCED ENGLISH LOGIC
    missing = []
    text_lower = text.lower()
    
    # 1. Contact Info
    if "@" not in text and not any(c.isdigit() for c in text): 
        missing.append("Contact Info (Email/Phone)")
        
    # 2. Education
    edu_keywords = ["education", "university", "degree", "college", "school", "academic", "bachelor", "master", "phd", "bsc", "msc"]
    if not any(kw in text_lower for kw in edu_keywords):
        missing.append("Education")
        
    # 3. Experience
    exp_keywords = ["experience", "work", "employment", "project", "activity", "internship", "skills", "summary", "objective", "professional"]
    if not any(kw in text_lower for kw in exp_keywords):
        missing.append("Experience")
    
    return missing, "English"

def parse_question_content(raw_text):
    lines = raw_text.split('\n')
    question_lines = []
    options = []
    option_pattern = re.compile(r'^\s*[A-D][\.\)]\s+')
    
    for line in lines:
        if option_pattern.match(line):
            options.append(line.strip())
        elif options: 
            options[-1] += " " + line.strip()
        else:
            question_lines.append(line)
            
    question_text = "\n".join(question_lines).strip()
    return (question_text, options) if len(options) >= 2 else (raw_text, None)

def evaluate_interview(provider, api_key, cv_text, q_a_history, position, company_name, custom_jd):
    """
    Evaluates the interview based on the Custom JD and CV.
    """
    jd_context = f"CUSTOM JOB DESCRIPTION:\n{custom_jd}" if custom_jd else f"STANDARD ROLE: {position}"
    
    prompt = f"""
    You are James, a strict Technical Hiring Manager at {company_name}.
    
    {jd_context}
    
    CANDIDATE CV (Excerpt):
    {cv_text[:1500]}...
    
    INTERVIEW HISTORY:
    --- SPECIALIZED ---
    {json.dumps(q_a_history.get('specialized', []), indent=2)}
    --- BEHAVIORAL ---
    {json.dumps(q_a_history.get('attitude', []), indent=2)}
    --- CODING ---
    {json.dumps(q_a_history.get('coding', []), indent=2)}
    
    TASK:
    1. Grade Specialized Knowledge (Count correct answers based on the JD).
    2. Grade Attitude (Professionalism & Cultural Fit).
    3. Grade Coding (Logic & Efficiency).
    4. Rate CV quality (0-10).
    
    OUTPUT JSON format ONLY:
    {{
        "specialized_correct_count": <int>,
        "attitude_accepted_count": <int>,
        "coding_accepted_count": <int>,
        "cv_score": <float>,
        "feedback_markdown": "Markdown string (English)"
    }}
    
    MARKDOWN STRUCTURE:
    - **Decision:** HIRE / NO HIRE
    - **Analysis:**
        * **CV:** ...
        * **Tech:** ...
        * **Behavior:** ...
    - **James's Advice:** Actionable steps to improve.
    """
    
    response = call_llm(provider, "gemini-2.0-flash-exp", api_key, prompt)
    
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        data = json.loads(json_match.group(0)) if json_match else json.loads(response)
        
        # Calculate Scores (0-10 scale)
        total_spec = len(q_a_history.get('specialized', [])) or 1
        spec_score = (data.get('specialized_correct_count', 0) / total_spec) * 10
        
        total_att = len(q_a_history.get('attitude', [])) or 1
        att_score = (data.get('attitude_accepted_count', 0) / total_att) * 10
        
        total_code = len(q_a_history.get('coding', [])) or 1
        think_score = (data.get('coding_accepted_count', 0) / total_code) * 10
        
        avg_score = (spec_score + att_score + think_score + data.get('cv_score', 0)) / 4
        status = "HIRED" if avg_score >= 7.0 else "NOT HIRED"
        
        return {
            "cv_score": round(data.get('cv_score', 0), 1),
            "specialized_score": round(spec_score, 1),
            "attitude_score": round(att_score, 1),
            "thinking_score": round(think_score, 1),
            "status": status,
            "feedback_markdown": data.get('feedback_markdown', 'No feedback.')
        }
    except:
        return {"cv_score": 0, "status": "ERROR", "feedback_markdown": "Error parsing AI response."}

def timer_component(minutes, key_suffix):
    seconds = minutes * 60
    timer_html = f"""
    <script>
    var timeLeft = {seconds};
    var elem = document.getElementById("timer_display_{key_suffix}");
    var timerId = setInterval(countdown, 1000);
    function countdown() {{
      if (timeLeft == -1) {{ clearTimeout(timerId); }} else {{
        var m = Math.floor(timeLeft / 60);
        var s = timeLeft % 60;
        if (s < 10) s = '0' + s;
        var displayString = m + ':' + s;
        var timerDiv = window.parent.document.getElementById("custom_timer_div");
        if (timerDiv) {{
             timerDiv.innerHTML = "⏱️ Time Left: " + displayString;
             if (timeLeft < 30) {{ timerDiv.style.backgroundColor = "#b91c1c"; }}
        }}
        timeLeft--;
      }}
    }}
    </script>
    """
    st.components.v1.html(timer_html, height=0)

# --- Session State ---
if 'step' not in st.session_state: st.session_state.step = 'setup'
if 'history' not in st.session_state: st.session_state.history = {'specialized': [], 'attitude': [], 'coding': []}
if 'target_company' not in st.session_state: st.session_state.target_company = "Tech Corp Inc."

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("Application Settings")
    uploaded_file = st.file_uploader("Upload CV/Resume (PDF, DOCX, Image)", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        st.success("CV Uploaded")
        st.markdown("---")
        st.header("Job Context")
        
        # FIX: Force Google Gemini and use Global API Key
        provider = "Google Gemini"
        user_api_key = API_KEY
        
        demo_mode = st.toggle("⚡ Demo mode (3 Questions)", value=True)
        
        position = st.selectbox("Job Position", [
            "Frontend Developer", "Backend Developer", "Fullstack Developer", "Data Scientist", 
            "DevOps Engineer", "Business Analyst", "Cybersecurity", "AI Engineer"
        ], index=0)
        
        experience = st.selectbox("Experience Level", [
            "Intern", "Fresher", "Junior", "Mid-Level", "Senior", "Lead"
        ])
        
        # CUSTOM JD INPUT
        custom_jd = st.text_area("Paste Job Description (JD)", height=150, placeholder="Paste the real JD here for tailored questions...")
        
        st.info(f"Target: {position} at {st.session_state.target_company}")
        
        col1, col2 = st.columns(2)
        start_btn = col1.button("START", type="primary")
        reset_btn = col2.button("RESET", disabled=(st.session_state.step == 'setup'))

# --- MAIN LOGIC ---

if 'reset_btn' in locals() and reset_btn:
    for key in st.session_state.keys():
        if key != 'target_company': del st.session_state[key]
    st.rerun()

if 'start_btn' in locals() and start_btn:
    if not uploaded_file:
        st.error("⚠️ Please upload a CV first.")
    elif not API_KEY:
        st.error("⚠️ API Key missing. Check secrets.")
    else:
        text_result = process_uploaded_file(uploaded_file, provider, user_api_key)
        if not text_result or text_result.startswith("Error"):
            st.error(f"File Error: {text_result}")
        else:
            st.session_state.resume_text = text_result
            st.session_state.custom_jd = custom_jd # Save JD to session
            st.session_state.step = 'cv_review'
            st.rerun()

# SCREENS
if st.session_state.step == 'setup':
    st.title("James - AI Interviewer Pro")
    st.markdown("""
    ### 👋 Welcome! I'm James, your personal AI Interviewer.
    
    I will conduct a mock interview based on your **CV** and the **Job Description** you provide.
    
    **How it works:**
    1.  **Upload CV:** I'll analyze your background.
    2.  **Paste JD:** (Optional) I'll ask specific questions matching the real job requirements.
    3.  **Interview:** We'll go through Technical, Behavioral, and Coding rounds.
    """)

elif st.session_state.step == 'cv_review':
    wait_time = 1 if demo_mode else 3
    placeholder = st.empty()
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        if elapsed >= (wait_time * 60): break
        remaining = int((wait_time * 60) - elapsed)
        mins, secs = divmod(remaining, 60)
        
        placeholder.markdown(f"""
        <div class="processing-overlay">
            <div style="font-size: 80px;">📄</div>
            <h2>James is reviewing your CV against the JD...</h2>
            <p>Time remaining: {mins:02d}:{secs:02d}</p>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1)
    
    placeholder.empty()
    missing, lang = check_cv_elements(st.session_state.resume_text)
    
    if missing and not demo_mode:
        st.error(f"❌ CV Rejected. Missing: {', '.join(missing)}")
        st.stop()
    else:
        st.success("✅ CV Accepted. Starting Interview...")
        time.sleep(1)
        st.session_state.step = 'specialized_intro'
        st.rerun()

elif st.session_state.step == 'specialized_intro':
    st.title("🧠 Phase 1: Technical Knowledge")
    st.info(f"James will now ask technical questions based on your CV and the JD for **{position}**.")
    if st.button("I'm Ready"):
        st.session_state.q_count = 3 if demo_mode else 5
        st.session_state.current_q_idx = 0
        st.session_state.step = 'specialized_questions'
        st.rerun()

elif st.session_state.step == 'specialized_questions':
    if st.session_state.current_q_idx < st.session_state.q_count:
        q_num = st.session_state.current_q_idx + 1
        
        if f"q_spec_{q_num}" not in st.session_state:
            with st.spinner("James is formulating a question..."):
                jd_context = f"Based on this JD: {st.session_state.custom_jd}" if st.session_state.custom_jd else ""
                prompt = f"""
                You are James, a technical interviewer.
                {jd_context}
                Generate a Multiple Choice Question (A,B,C,D) for a {position} ({experience}).
                Focus on requirements found in the JD.
                Format: Question text followed by options.
                """
                q_text = call_llm(provider, "gemini-2.0-flash-exp", user_api_key, prompt)
                st.session_state[f"q_spec_{q_num}"] = q_text
        
        st.markdown(f'<div id="custom_timer_div" class="timer-display">⏱️ Time Left: 2:00</div>', unsafe_allow_html=True)
        timer_component(2, f"spec_{q_num}")
        
        q_content, options = parse_question_content(st.session_state[f"q_spec_{q_num}"])
        st.subheader(f"Question {q_num}")
        st.write(q_content)
        
        answer = st.radio("Select:", options, key=f"ans_spec_{q_num}", index=None) if options else st.text_area("Answer:", key=f"ans_spec_{q_num}")
        
        if st.button("Next"):
            if answer:
                st.session_state.history['specialized'].append({"question": q_content, "answer": answer})
                st.session_state.current_q_idx += 1
                st.rerun()
    else:
        st.session_state.step = 'attitude_intro'
        st.rerun()

elif st.session_state.step == 'attitude_intro':
    st.title("🤝 Phase 2: Behavioral Fit")
    if st.button("Start Behavioral Round"):
        st.session_state.q_count_att = 2 if demo_mode else 4
        st.session_state.current_q_att_idx = 0
        st.session_state.step = 'attitude_questions'
        st.rerun()

elif st.session_state.step == 'attitude_questions':
    if st.session_state.current_q_att_idx < st.session_state.q_count_att:
        q_num = st.session_state.current_q_att_idx + 1
        if f"q_att_{q_num}" not in st.session_state:
            with st.spinner("James is thinking..."):
                prompt = f"Generate a Behavioral Question (Situational) for {position}. Multiple Choice format."
                st.session_state[f"q_att_{q_num}"] = call_llm(provider, "gemini-2.0-flash-exp", user_api_key, prompt)
        
        st.markdown(f'<div id="custom_timer_div" class="timer-display">⏱️ Time Left: 3:00</div>', unsafe_allow_html=True)
        timer_component(3, f"att_{q_num}")
        
        q_content, options = parse_question_content(st.session_state[f"q_att_{q_num}"])
        st.subheader(f"Question {q_num}")
        st.write(q_content)
        answer = st.radio("Select:", options, key=f"ans_att_{q_num}", index=None) if options else st.text_area("Answer:", key=f"ans_att_{q_num}")
        
        if st.button("Next"):
            if answer:
                st.session_state.history['attitude'].append({"question": q_content, "answer": answer})
                st.session_state.current_q_att_idx += 1
                st.rerun()
    else:
        st.session_state.step = 'evaluation'
        st.rerun()

elif st.session_state.step == 'evaluation':
    st.title("📊 Final Report from James")
    
    if 'eval_complete' not in st.session_state:
        with st.spinner("James is grading your answers against the JD..."):
            res = evaluate_interview(
                provider, user_api_key, 
                st.session_state.resume_text, 
                st.session_state.history, 
                position, 
                st.session_state.target_company,
                st.session_state.custom_jd
            )
            st.session_state.scores = res
            st.session_state.eval_complete = True
            st.rerun()
    
    scores = st.session_state.scores
    color = "#22c55e" if scores['status'] == "HIRED" else "#ef4444"
    
    st.markdown(f"""
    <div style="background-color: #1e293b; padding: 30px; border-radius: 15px; text-align: center; border: 2px solid {color};">
        <h1 style="color: {color} !important; margin: 0;">{scores['status']}</h1>
        <h3 style="color: #cbd5e1;">Resume Score: {scores['cv_score']}/10</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Technical", f"{scores['specialized_score']}/10")
    col2.metric("Behavioral", f"{scores['attitude_score']}/10")
    col3.metric("Coding Logic", f"{scores['thinking_score']}/10")
    
    st.markdown("### 📝 James's Feedback")
    st.markdown(scores['feedback_markdown'])
    
    if st.button("Start New Interview"):
        for key in st.session_state.keys(): del st.session_state[key]
        st.rerun()


