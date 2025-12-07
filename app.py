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

# --- Configuration & Setup ---
st.set_page_config(page_title="AI Tech Interviewer Pro", layout="wide")

# --- Custom CSS for Night Blue Theme & Animations ---
st.markdown("""
    <style>
    /* Main Background - Deep Night Blue */
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    
    /* Sidebar - Slightly Lighter Night Blue */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
    }
    
    /* Inputs */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #334155;
        color: #f8fafc;
        border-color: #475569;
    }
    
    /* Selectbox */
    .stSelectbox > div > div > div {
        background-color: #334155;
        color: #f8fafc;
    }
    
    /* Start/Reset Buttons */
    .stButton > button {
        background-color: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }
    
    /* Custom Timer Logic */
    .timer-display {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: #dc2626;
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: bold;
        z-index: 9999;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        font-family: monospace;
        font-size: 18px;
    }
    
    /* Processing Overlay */
    .processing-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: rgba(15, 23, 42, 0.95);
        z-index: 999999;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: white;
    }
    
    /* Radio Button Text Color */
    .stRadio label {
        color: #e2e8f0 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper to get keys from secrets/env ---
def get_key(name):
    return st.secrets.get(name, os.getenv(name))

# --- Universal LLM Caller Function (Fixed Models & Image Support & Auto-Fallback) ---
def call_llm(provider, model_name, api_key, prompt, image_data=None, retries=2):
    """
    Handles Text AND Image inputs for OCR and Interviewing.
    Includes auto-fallback to other providers if the primary one fails.
    """
    
    # 1. Attempt Primary Provider
    result = _try_provider(provider, model_name, api_key, prompt, image_data)
    if result and not result.startswith("Error"):
        return result
    
    # 2. Auto-Fallback Logic
    print(f"Primary provider {provider} failed. Attempting fallbacks...")
    
    # Define fallback priority (check if keys exist)
    fallbacks = []
    
    # Add Gemini as fallback
    gemini_key = get_key("GEMINI_API_KEY")
    if gemini_key and provider != "Google Gemini":
        fallbacks.append(("Google Gemini", "gemini-2.0-flash-exp", gemini_key))
        
    # Add Groq as fallback (Text only)
    groq_key = get_key("GROQ_API_KEY")
    if groq_key and provider != "Groq" and not image_data:
        fallbacks.append(("Groq", "llama-3.3-70b-versatile", groq_key))
        
    # Add OpenAI as fallback
    openai_key = get_key("OPENAI_API_KEY")
    if openai_key and provider != "OpenAI":
        fallbacks.append(("OpenAI", "gpt-4o", openai_key))

    # Attempt fallbacks
    for fb_provider, fb_model, fb_key in fallbacks:
        res = _try_provider(fb_provider, fb_model, fb_key, prompt, image_data)
        if res and not res.startswith("Error"):
            return res
            
    return f"Error: All available AI models failed. Primary Error: {result}"

def _try_provider(provider, model_name, api_key, prompt, image_data):
    if not api_key:
        return "Error: API Key missing."

    # --- Google Gemini ---
    if provider == "Google Gemini":
        try:
            genai.configure(api_key=api_key)
            # Updated Robust fallback list for Gemini
            candidates = [model_name, 'gemini-2.0-flash-exp', 'gemini-1.5-flash-latest', 'gemini-1.5-flash', 'gemini-1.5-pro']
            
            for m in candidates:
                try:
                    model = genai.GenerativeModel(m)
                    if image_data:
                        response = model.generate_content([prompt, image_data])
                    else:
                        response = model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    continue
            return "Error: All Gemini models busy/not found."
        except Exception as e:
            return f"Gemini Error: {str(e)}"

    # --- Groq (Text Only) ---
    elif provider == "Groq":
        if image_data:
            return "Error: Groq does not support image input."
            
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
            
            # Updated active Groq models
            candidates = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
            
            for m in candidates:
                try:
                    chat = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt}],
                        model=m,
                    )
                    return chat.choices[0].message.content
                except Exception:
                    continue
            return "Error: Groq models unavailable."
        except Exception as e:
            return f"Groq Error: {str(e)}"

    # --- OpenAI ---
    elif provider == "OpenAI":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            if image_data:
                # Convert PIL Image to Base64
                buffered = io.BytesIO()
                image_data.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                        ]
                    }
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI Error: {str(e)}"

    # --- Claude ---
    elif provider == "Anthropic (Claude)":
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            
            if image_data:
                # Convert PIL Image to Base64
                buffered = io.BytesIO()
                image_data.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_str}},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            message = client.messages.create(
                model=model_name,
                max_tokens=1024,
                messages=messages
            )
            return message.content[0].text
        except Exception as e:
            return f"Claude Error: {str(e)}"
            
    return "Error: Unknown provider"

# --- Helper Functions ---

def extract_text_from_docx(file):
    try:
        import docx
        doc = docx.Document(file)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return '\n'.join(text)
    except ImportError:
        return "Error: python-docx library not installed. Please install it to read .docx files."
    except Exception as e:
        return f"Error reading .docx file: {e}"

def process_uploaded_file(uploaded_file, provider, user_api_key):
    """
    Extracts text from PDF, DOCX, or Image (OCR via LLM).
    Returns: extracted_text (str)
    """
    try:
        # 1. Handle PDF
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return " ".join(text.split())

        # 2. Handle DOCX
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_text_from_docx(uploaded_file)

        # 3. Handle Images (OCR)
        elif uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
            image = Image.open(uploaded_file)
            
            # OCR Prompt
            prompt = "Transcribe the text from this CV/Resume image exactly as it appears. Structure it clearly."
            
            # Special Handling for Groq (No Vision) -> Fallback to Gemini if possible
            ocr_provider = provider
            ocr_key = user_api_key
            # Updated default fallback model to 2.0-flash-exp (Fixes 404 on 1.5-flash)
            ocr_model = "gemini-2.0-flash-exp" 
            
            if provider == "Groq":
                # Check for Gemini key in secrets for fallback
                gemini_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
                if gemini_key:
                    ocr_provider = "Google Gemini"
                    ocr_key = gemini_key
                else:
                    # Try OpenAI if Gemini missing
                    openai_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
                    if openai_key:
                        ocr_provider = "OpenAI"
                        ocr_key = openai_key
                        ocr_model = "gpt-4o"
                    else:
                        return "Error: Groq cannot read images and no Gemini/OpenAI fallback key found. Please upload a PDF."

            with st.spinner(f"Reading image using {ocr_provider}..."):
                text = call_llm(ocr_provider, ocr_model, ocr_key, prompt, image_data=image)
            
            return text

    except Exception as e:
        return f"Error processing file: {e}"
    return None

def get_job_description(position, company):
    return f"""
    ### {company} Recruitment
    **Role:** {position}
    **Key Requirements:**
    * Deep understanding of {position} core concepts.
    * Experience with scalable systems and performance optimization.
    * Proficiency in relevant languages (Python/C++/Java).
    * Strong problem-solving skills and teamwork.
    * Willingness to learn and adapt to new technologies.
    """

def get_future_date():
    days_ahead = random.randint(7, 30)
    future_date = datetime.now() + timedelta(days=days_ahead)
    return future_date.strftime("%B %d, %Y")

@st.dialog("Explore Opportunities")
def show_opportunities_popup():
    st.markdown("### Choose your next interview:")
    
    # Generate 3 random companies distinct from current if possible
    all_companies = ["Google", "Amazon", "Microsoft", "Netflix", "Tesla", "SpaceX", "Adobe", "Apple", "Meta"]
    current = st.session_state.get('target_company', '')
    candidates = [c for c in all_companies if c != current]
    options = random.sample(candidates, 3)
    
    selected_company = st.radio("Select a company:", options)
    
    if st.button("Confirm Selection"):
        st.session_state.target_company = selected_company
        # Reset Logic
        for key in st.session_state.keys():
            if key != 'target_company': # Keep the new company selection
                del st.session_state[key]
        st.rerun()

def check_cv_elements(text):
    # Determine Language first, then check criteria
    missing = []
    text_lower = text.lower()
    
    # Simple heuristic to detect Vietnamese: check for specific characters
    # chars like: ư, ơ, đ, and accents
    vi_chars = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
    vi_char_count = sum(1 for char in text_lower if char in vi_chars)
    
    # Threshold to decide language (e.g., if > 3 Vietnamese characters found)
    is_vietnamese = vi_char_count > 3
    detected_language = "Vietnamese" if is_vietnamese else "English"
    
    # 1. Contact Info (Universal - Email or Phone)
    if "@" not in text and not any(c.isdigit() for c in text): 
        missing.append("Contact Info (Email/Phone)")
        
    if is_vietnamese:
        # --- VIETNAMESE CRITERIA ---
        # 2. Education
        edu_keywords = ["học vấn", "trường", "đại học", "cao đẳng", "bằng cấp", "giáo dục", "chứng chỉ"]
        if not any(kw in text_lower for kw in edu_keywords):
            missing.append("Học vấn (Education)")
            
        # 3. Experience (Expanded for Students/Freshers)
        exp_keywords = [
            "kinh nghiệm", "làm việc", "dự án", "hoạt động", "sản phẩm", "thực tập",
            "mục tiêu", "kỹ năng", "ưu điểm", "thành tích", "giới thiệu"
        ]
        if not any(kw in text_lower for kw in exp_keywords):
            missing.append("Kinh nghiệm (Experience)")
            
    else:
        # --- ENGLISH CRITERIA ---
        # 2. Education
        edu_keywords = ["education", "university", "degree", "college", "school", "academic"]
        if not any(kw in text_lower for kw in edu_keywords):
            missing.append("Education")
            
        # 3. Experience (Expanded for Students)
        exp_keywords = ["experience", "work", "employment", "project", "activity", "internship", "skills", "summary", "objective"]
        if not any(kw in text_lower for kw in exp_keywords):
            missing.append("Experience")
    
    return missing, detected_language

def parse_question_content(raw_text):
    """
    Parses LLM output to separate Question Text from Options (A, B, C, D).
    Returns: (question_text, options_list)
    """
    # Try to find options pattern like "A) " or "A. "
    # If found, split. If not, return raw text and empty options (implies text input needed)
    
    # Basic Split by newline to find options
    lines = raw_text.split('\n')
    question_lines = []
    options = []
    
    # Regex for options A. or A)
    option_pattern = re.compile(r'^\s*[A-D][\.\)]\s+')
    
    for line in lines:
        if option_pattern.match(line):
            options.append(line.strip())
        elif options: 
            # If we already found options, append to last option (multiline option)
            options[-1] += " " + line.strip()
        else:
            question_lines.append(line)
            
    question_text = "\n".join(question_lines).strip()
    
    if len(options) >= 2:
        return question_text, options
    else:
        # Fallback if AI didn't format as MC properly
        return raw_text, None

def evaluate_interview(provider, api_key, cv_text, q_a_history, position, company_name):
    """
    Uses LLM to grade the entire interview based on correct answer percentages.
    Converts counts to Vietnam 10-point scale.
    """
    prompt = f"""
    You are a strict technical interviewer evaluating a candidate for the position of {position} at {company_name}.
    
    Here is the Candidate's CV Content (Excerpt):
    {cv_text[:1500]}...
    
    Here are the Interview Questions and the Candidate's Answers:
    
    --- SPECIALIZED KNOWLEDGE ---
    {json.dumps(q_a_history.get('specialized', []), indent=2)}
    
    --- ATTITUDE & BEHAVIORAL ---
    {json.dumps(q_a_history.get('attitude', []), indent=2)}
    
    --- CODING CHALLENGE ---
    {json.dumps(q_a_history.get('coding', []), indent=2)}
    
    INSTRUCTIONS:
    1. **Specialized Knowledge:** Count exactly how many answers are factually correct.
    2. **Attitude:** Count exactly how many answers are professional and acceptable. Any answer that is nonsensical, gibberish, rude, or toxic must be REJECTED (not counted).
    3. **Coding:** Count exactly how many solutions are logically correct and solve the problem.
    4. **CV/Resume:** Rate the CV quality on a scale of 0 to 10 (Float).
    5. Return ONLY a JSON object with this EXACT structure (no markdown formatting around it):
    {{
        "specialized_correct_count": <int>,
        "attitude_accepted_count": <int>,
        "coding_accepted_count": <int>,
        "cv_score": <float>,
        "feedback_markdown": "Markdown string strictly following the requested format."
    }}
    
    FORMAT FOR 'feedback_markdown':
    - **Company Name:** {company_name}
    - **Feedback:**
        * **CV:** [Detailed analysis of CV strengths/weaknesses]
        * **Specialized:** [Analysis of technical answers]
        * **Attitude:** [Analysis of behavioral answers]
        * **Coding:** [Analysis of code quality/logic]
    - **Suggestions:** [Comprehensive strategies to improve based on all the above]
    
    Make the Suggestions section long, comprehensive, and actionable.
    """
    
    response = call_llm(provider, "llama-3.3-70b-versatile", api_key, prompt)
    
    # JSON Parsing Fallback
    try:
        # Attempt to find JSON block in case of conversational wrapper
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            data = json.loads(response)
            
        # --- Python-side Calculation for Vietnam Grading Scale (0-10) ---
        # Score = (Correct Answers / Total Questions) * 10
        
        # Specialized
        total_spec = len(q_a_history.get('specialized', []))
        spec_score = (data.get('specialized_correct_count', 0) / total_spec * 10) if total_spec > 0 else 0
        
        # Attitude
        total_att = len(q_a_history.get('attitude', []))
        att_score = (data.get('attitude_accepted_count', 0) / total_att * 10) if total_att > 0 else 0
        
        # Coding (Thinking)
        total_code = len(q_a_history.get('coding', []))
        think_score = (data.get('coding_accepted_count', 0) / total_code * 10) if total_code > 0 else 0
        
        # CV Score comes directly from AI (subjective quality)
        cv_score = data.get('cv_score', 0)
        
        # Determine Status (Average >= 7.0 is HIRED)
        avg_score = (spec_score + att_score + think_score + cv_score) / 4
        status = "HIRED" if avg_score >= 7.0 else "NOT HIRED"
        
        return {
            "cv_score": round(cv_score, 1),
            "specialized_score": round(spec_score, 1),
            "attitude_score": round(att_score, 1),
            "thinking_score": round(think_score, 1),
            "status": status,
            "feedback_markdown": data.get('feedback_markdown', 'No feedback provided.')
        }
        
    except Exception as e:
        # Default fail-safe score
        return {
            "cv_score": 0, "specialized_score": 0, "attitude_score": 0, "thinking_score": 0,
            "status": "NOT HIRED", 
            "feedback_markdown": f"Error parsing AI evaluation: {e}"
        }

# --- Javascript Timer Component ---
def timer_component(minutes, key_suffix):
    seconds = minutes * 60
    # CSS/JS to show a countdown in the bottom right
    timer_html = f"""
    <script>
    var timeLeft = {seconds};
    var elem = document.getElementById("timer_display_{key_suffix}");
    var timerId = setInterval(countdown, 1000);
    
    function countdown() {{
      if (timeLeft == -1) {{
        clearTimeout(timerId);
        // Optional: Trigger streamit rerun logic via hidden button click if needed
      }} else {{
        var m = Math.floor(timeLeft / 60);
        var s = timeLeft % 60;
        if (s < 10) s = '0' + s;
        var displayString = m + ':' + s;
        
        // Update the existing div created by Streamlit markdown
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

# --- Session State Initialization ---
if 'step' not in st.session_state: st.session_state.step = 'setup'
if 'scores' not in st.session_state: st.session_state.scores = {'cv': 0, 'specialized': 0, 'attitude': 0, 'thinking': 0}
if 'history' not in st.session_state: st.session_state.history = {'specialized': [], 'attitude': [], 'coding': []}

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("Application")
    # 1. CV Upload (Updated for Images and DOCX)
    uploaded_file = st.file_uploader("Upload CV/Resume (PDF, DOCX, Image)", type=['pdf', 'docx', 'png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        st.success("File Uploaded Successfully")
        st.markdown("---")
        
        st.header("Job Information")
        
        # Auto-API Logic (Hidden)
        provider = "Groq" # Default free
        user_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
        if not user_api_key:
            provider = "Google Gemini"
            user_api_key = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
        
        # 2. Toggle Switch for Demo Mode
        demo_mode = st.toggle("⚡ Demo mode (3 questions)", value=True)
        
        # 3. Job Position (Alphabetical)
        job_list = sorted([
            "Artificial Intelligence Engineer", 
            "Backend Developer", 
            "Business Analyst",
            "Cybersecurity", 
            "Data Scientist", 
            "DevOps Engineer", 
            "Frontend Developer", 
            "Fullstack Developer", 
            "Human Resource Manager",
            "Software Developer"
        ])
        
        position = st.selectbox("Job Position", job_list, index=None, placeholder="Select a position...")
        
        # Only show details if position is selected
        if position:
            # Persistent Company Assignment
            if 'target_company' not in st.session_state:
                st.session_state.target_company = random.choice(["NVIDIA", "Intel", "IBM", "AMD", "Facebook"])
            
            # 4. Experience
            experience = st.selectbox("Experience", [
                "Fresher", "Intern", "Junior", "Mid-Level", "Senior", "Lead/Manager"
            ])
            
            # 5. Job Description
            st.markdown("### Job Description")
            st.write(f"**Applying to:** {st.session_state.target_company}")
            jd_text = get_job_description(position, st.session_state.target_company)
            st.info(jd_text)
            
            # 6. Buttons
            col1, col2 = st.columns(2)
            start_btn = col1.button("START", type="primary")
            reset_btn = col2.button("RESET", disabled=(st.session_state.step == 'setup'))

# --- MAIN LOGIC FLOW ---

# Handle Reset
if 'reset_btn' in locals() and reset_btn:
    # Clear session state keys but NOT the API keys if they exist in environment
    # We actually want to clear target_company on explicit reset to allow re-roll or fresh start
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

# Handle Start
if 'start_btn' in locals() and start_btn:
    if not uploaded_file:
        st.error("⚠️ Please upload a CV first.")
    else:
        # Unified Processing for PDF, DOCX and Images
        text_result = process_uploaded_file(uploaded_file, provider, user_api_key)
        
        if not text_result or text_result.startswith("Error"):
            st.error(f"Failed to read file: {text_result}")
        else:
            st.session_state.resume_text = text_result
            st.session_state.step = 'cv_review'
            st.rerun()

# Default Screen
if st.session_state.step == 'setup':
    st.title("Welcome to the AI Recruitment Portal")
    st.markdown("""
    Please upload your CV in the sidebar and select a job position to begin.
    
    **Instructions:**
    1.  **Application:** Upload your CV (PDF, DOCX or Image).
    2.  **Job Information:** Select your role and experience.
    3.  **Interview:**
        * **CV Review:** We will analyze your document.
        * **Specialized Knowledge:** Technical questions.
        * **Attitude:** Behavioral questions.
        * **Coding:** Programming challenge (Upload code file).
    """)

# --- STEP 1: CV REVIEW (Wait Screen) ---
elif st.session_state.step == 'cv_review':
    # Calculate Wait Time
    # Standard: 3 mins. Demo Mode (Reduce by 2/3): 1 min.
    cv_wait_time = 1 if demo_mode else 3
    
    placeholder = st.empty()
    start_time = time.time()
    total_seconds = cv_wait_time * 60
    
    # Real-time countdown loop
    while True:
        elapsed = time.time() - start_time
        if elapsed >= total_seconds:
            break
            
        remaining = int(total_seconds - elapsed)
        mins, secs = divmod(remaining, 60)
        time_str = f"{mins:02d}:{secs:02d}"
        
        # Calculate progress percentage (0 to 100)
        progress_pct = min((elapsed / total_seconds) * 100, 100)
        
        placeholder.markdown(f"""
        <div class="processing-overlay">
            <div style="font-size: 80px;">🕒</div>
            <h2>We are examining your CV/Resume.</h2>
            <p>Please come back after {cv_wait_time} minute(s).</p>
            <p style="font-size: 24px; font-weight: bold; margin-top: 10px;">{time_str}</p>
            <div style="margin-top: 20px; width: 300px; height: 10px; background: #334155; border-radius: 5px;">
                <div style="width: {progress_pct}%; height: 100%; background: #3b82f6; border-radius: 5px; transition: width 1s linear;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(1)
        
    placeholder.empty()
    
    # Criteria Check
    missing_elements, detected_lang = check_cv_elements(st.session_state.resume_text)
    
    if missing_elements and not demo_mode:
        st.error(f"❌ Application Rejected. Missing required elements: {', '.join(missing_elements)}")
        st.caption(f"Detected Language: {detected_lang}")
        
        with st.expander("Debug: View Extracted Text (For verification)"):
            st.text(st.session_state.resume_text[:2000] + "...")
            
        st.markdown("**Score: 0/10**")
        if st.button("Try Again"):
            st.session_state.step = 'setup'
            st.rerun()
        st.stop()
    else:
        # Proceed
        if demo_mode and missing_elements:
            st.warning("⚠️ Demo Mode Active: CV Criteria Checks are Bypassed.")
            time.sleep(1.5)
            
        st.success(f"✅ Application Accepted ({detected_lang})")
        time.sleep(1.5)
        st.session_state.scores['cv'] = 8.5 # Simulated score (will be updated by AI later)
        st.session_state.step = 'specialized_intro'
        st.rerun()

# --- STEP 2: SPECIALIZED KNOWLEDGE ---
elif st.session_state.step == 'specialized_intro':
    st.title("🧠 Phase 1: Specialized Knowledge")
    st.markdown(f"**Role:** {position} | **Experience:** {experience}")
    st.info("Purpose: Evaluate specialized knowledge for the selected job.")
    
    st.write("Sources: *Cracking the Coding Interview*, *Elements of Programming Interviews*")
    
    if st.button("Start Specialized Questions"):
        # Set question count based on Demo Mode
        st.session_state.q_count = 3 if demo_mode else 60
        st.session_state.current_q_idx = 0
        st.session_state.step = 'specialized_questions'
        st.rerun()

elif st.session_state.step == 'specialized_questions':
    if st.session_state.current_q_idx < st.session_state.q_count:
        q_num = st.session_state.current_q_idx + 1
        
        # Calculate Timer: Standard 2 mins. Demo (Reduce by 2/3) -> 0.66 mins (~40s)
        spec_time = (2/3) if demo_mode else 2
        spec_time_str = "40s" if demo_mode else "2:00"
        
        # Generate Question
        if f"q_spec_{q_num}" not in st.session_state:
            with st.spinner(f"Generating Technical Question {q_num}..."):
                # Prepare a context of previous questions to ensure uniqueness
                prev_questions = [item['question'] for item in st.session_state.history.get('specialized', [])]
                prev_q_text = " ".join(prev_questions)
                
                if demo_mode:
                    prompt = f"""Generate a UNIQUE, SIMPLE, BEGINNER-FRIENDLY technical interview question (Multiple Choice with 4 options A, B, C, D) for a {position} ({experience} level). 
                    Keep it short and easy. 
                    Ensure this question is DIFFERENT from these previous ones: {prev_q_text}
                    Format: Question text followed by A) Option B) Option... Source: Basic Programming Concepts."""
                else:
                    prompt = f"""Generate a UNIQUE, CHALLENGING, IN-DEPTH technical interview question (Multiple Choice with 4 options A, B, C, D) for a {position} ({experience} level). 
                    Focus on core concepts. 
                    Ensure this question is DIFFERENT from these previous ones: {prev_q_text}
                    Format: Question text followed by A) Option B) Option... Source: Cracking the Coding Interview."""
                
                q_text = call_llm(provider, "llama-3.3-70b-versatile", user_api_key, prompt)
                st.session_state[f"q_spec_{q_num}"] = q_text
        
        # Show Timer AFTER content is loaded
        st.markdown(f'<div id="custom_timer_div" class="timer-display">⏱️ Time Left: {spec_time_str}</div>', unsafe_allow_html=True)
        timer_component(spec_time, f"spec_{q_num}")
        
        # Parse Question Content
        q_content, options = parse_question_content(st.session_state[f"q_spec_{q_num}"])
        
        st.subheader(f"Question {q_num}/{st.session_state.q_count}")
        st.write(q_content)
        
        if options:
            answer = st.radio("Select an Answer:", options, key=f"ans_spec_{q_num}", index=None)
        else:
            answer = st.text_area("Your Answer:", key=f"ans_spec_{q_num}")
        
        if st.button("Next Question"):
            if not answer:
                st.warning("Please provide an answer.")
            else:
                # Save answer for grading
                st.session_state.history['specialized'].append({"question": q_content, "answer": answer})
                st.session_state.current_q_idx += 1
                st.rerun()
    else:
        st.session_state.step = 'attitude_intro'
        st.rerun()

# --- STEP 3: ATTITUDE ---
elif st.session_state.step == 'attitude_intro':
    st.title("🤝 Phase 2: Attitude & Behavioral")
    st.markdown("**Goal:** Assess work ethics and personality.")
    
    if st.button("Start Attitude Test"):
        # Reduced to 10 for Demo Mode (1/3 of 30)
        st.session_state.q_count_att = 10 if demo_mode else 30 
        st.session_state.current_q_att_idx = 0
        st.session_state.step = 'attitude_questions'
        st.rerun()

elif st.session_state.step == 'attitude_questions':
    if st.session_state.current_q_att_idx < st.session_state.q_count_att:
        q_num = st.session_state.current_q_att_idx + 1
        
        # Calculate Timer: Standard 10 mins. Demo (Reduce by 2/3) -> 3.33 mins
        att_time = (10/3) if demo_mode else 10
        att_time_str = "3:20" if demo_mode else "10:00"
        
        if f"q_att_{q_num}" not in st.session_state:
            with st.spinner(f"Generating Behavioral Question {q_num}..."):
                # Prepare context of previous questions
                prev_questions = [item['question'] for item in st.session_state.history.get('attitude', [])]
                prev_q_text = " ".join(prev_questions)
                
                if demo_mode:
                    prompt = f"""Generate a UNIQUE, SHORT, SIMPLE behavioral interview question #{q_num} (Teamwork/Conflict/Ethics). 
                    Ensure it is DIFFERENT from: {prev_q_text}.
                    Multiple Choice format."""
                else:
                    prompt = f"""Generate a UNIQUE, COMPLEX behavioral interview question #{q_num} (Teamwork/Conflict/Ethics). 
                    Ensure it is DIFFERENT from: {prev_q_text}.
                    Multiple Choice format."""
                    
                q_text = call_llm(provider, "llama-3.3-70b-versatile", user_api_key, prompt)
                st.session_state[f"q_att_{q_num}"] = q_text
            
        # Show Timer AFTER content is loaded
        st.markdown(f'<div id="custom_timer_div" class="timer-display">⏱️ Time Left: {att_time_str}</div>', unsafe_allow_html=True)
        timer_component(att_time, f"att_{q_num}")
        
        # Parse Question Content
        q_content, options = parse_question_content(st.session_state[f"q_att_{q_num}"])
        
        st.subheader(f"Behavioral Question {q_num}/{st.session_state.q_count_att}")
        st.write(q_content)
        
        if options:
            answer = st.radio("Select an Answer:", options, key=f"ans_att_{q_num}", index=None)
        else:
            answer = st.text_area("Your Answer:", key=f"ans_att_{q_num}")
        
        if st.button("Next"):
            # Save answer for grading
            st.session_state.history['attitude'].append({"question": q_content, "answer": answer})
            st.session_state.current_q_att_idx += 1
            st.rerun()
    else:
        st.session_state.step = 'coding_intro'
        st.rerun()

# --- STEP 4: CODING CHALLENGE ---
elif st.session_state.step == 'coding_intro':
    st.title("💻 Phase 3: Coding Challenge")
    st.markdown("**Goal:** Check clean code, thinking process, and algorithms.")
    st.markdown("**Allowed Languages:** Python, C++, Java, JavaScript, Go, Ruby, PHP, C#, Swift, Kotlin.")
    
    if st.button("Start Coding Challenge"):
        st.session_state.q_count_code = 1 if demo_mode else 3
        st.session_state.current_q_code_idx = 0
        st.session_state.step = 'coding_questions'
        st.rerun()

elif st.session_state.step == 'coding_questions':
    if st.session_state.current_q_code_idx < st.session_state.q_count_code:
        q_num = st.session_state.current_q_code_idx + 1
        
        # Calculate Timer: Standard 10 mins. Demo (Reduce by 2/3) -> 3.33 mins
        code_time = (10/3) if demo_mode else 10
        code_time_str = "3:20" if demo_mode else "10:00"
        
        if f"q_code_{q_num}" not in st.session_state:
            with st.spinner(f"Generating Coding Problem {q_num}..."):
                # Prepare context of previous questions
                prev_questions = [item['question'] for item in st.session_state.history.get('coding', [])]
                prev_q_text = " ".join(prev_questions)
                
                if demo_mode:
                    prompt = f"""Generate a UNIQUE, EASY coding algorithm problem (e.g., FizzBuzz, String Reversal) for a {position}. 
                    Ensure it is DIFFERENT from: {prev_q_text}.
                    Short problem statement."""
                else:
                    prompt = f"""Generate a UNIQUE, MEDIUM/HARD coding algorithm problem (e.g., Graphs, DP) for a {position}. 
                    Ensure it is DIFFERENT from: {prev_q_text}.
                    Detailed problem statement."""
                    
                q_text = call_llm(provider, "llama-3.3-70b-versatile", user_api_key, prompt)
                st.session_state[f"q_code_{q_num}"] = q_text
            
        # Show Timer AFTER content is loaded
        st.markdown(f'<div id="custom_timer_div" class="timer-display">⏱️ Time Left: {code_time_str}</div>', unsafe_allow_html=True)
        timer_component(code_time, f"code_{q_num}")
        
        st.subheader(f"Coding Problem {q_num}")
        st.info(st.session_state[f"q_code_{q_num}"])
        
        # Language Check
        language = st.selectbox("Select Language", 
            ["Python", "C++", "Java", "JavaScript", "Go", "Ruby", "PHP", "C#", "Swift", "Kotlin"], 
            key=f"lang_{q_num}")
        
        # File Upload for Code
        code_file = st.file_uploader("Upload Code File or Image", type=['py', 'cpp', 'java', 'js', 'go', 'rb', 'php', 'cs', 'swift', 'kt', 'png', 'jpg'], key=f"file_{q_num}")
        code_text_input = st.text_area("Or type code here:", height=200, key=f"text_{q_num}")
        
        if st.button("Submit Code"):
            if not code_file and not code_text_input:
                st.warning("Please upload a file or write code.")
            else:
                # Save answer
                ans_content = code_text_input if code_text_input else f"File uploaded: {code_file.name}"
                st.session_state.history['coding'].append({"question": st.session_state[f"q_code_{q_num}"], "answer": ans_content})
                
                st.success("Code received.")
                st.session_state.current_q_code_idx += 1
                st.rerun()
    else:
        st.session_state.step = 'evaluation'
        st.rerun()

# --- STEP 5: FINAL EVALUATION (UPDATED WITH REAL SCORING & TIMER) ---
elif st.session_state.step == 'evaluation':
    st.title("📊 Final Evaluation")
    
    # Calculate Evaluation Wait Time (10 mins in Full Mode, 3.33 mins in Demo)
    eval_wait_time = 3 # Fixed 3 minutes as requested
    
    # --- Real-Time Analysis Animation ---
    if 'eval_complete' not in st.session_state:
        placeholder = st.empty()
        start_time = time.time()
        total_seconds = eval_wait_time * 60
        
        while True:
            elapsed = time.time() - start_time
            if elapsed >= total_seconds:
                break
                
            remaining = int(total_seconds - elapsed)
            mins, secs = divmod(remaining, 60)
            time_str = f"{mins:02d}:{secs:02d}"
            
            # Calculate progress percentage (0 to 100)
            progress_pct = min((elapsed / total_seconds) * 100, 100)
            
            placeholder.markdown(f"""
            <div class="processing-overlay">
                <div style="font-size: 80px;">🧠</div>
                <h2>Analyzing Interview Performance...</h2>
                <p>Please wait while the AI board evaluates your answers.</p>
                <p style="font-size: 24px; font-weight: bold; margin-top: 10px;">{time_str}</p>
                <div style="margin-top: 20px; width: 300px; height: 10px; background: #334155; border-radius: 5px;">
                    <div style="width: {progress_pct}%; height: 100%; background: #10b981; border-radius: 5px; transition: width 1s linear;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)
            
        placeholder.empty()
        
        # --- Perform Real AI Scoring ---
        with st.spinner("Finalizing report..."):
            evaluation_result = evaluate_interview(
                provider, 
                user_api_key, 
                st.session_state.resume_text, 
                st.session_state.history,
                position,
                st.session_state.target_company
            )
            
            # Update Session State with Real Scores
            st.session_state.scores['cv'] = evaluation_result.get('cv_score', 0)
            st.session_state.scores['specialized'] = evaluation_result.get('specialized_score', 0)
            st.session_state.scores['attitude'] = evaluation_result.get('attitude_score', 0)
            st.session_state.scores['thinking'] = evaluation_result.get('thinking_score', 0)
            st.session_state.final_status = evaluation_result.get('status', 'NOT HIRED')
            st.session_state.final_feedback = evaluation_result.get('feedback_markdown', 'No feedback provided.')
            
            st.session_state.eval_complete = True
            st.rerun()

    # --- Display Results ---
    if st.session_state.get('eval_complete'):
        avg_score = sum(st.session_state.scores.values()) / 4
        status = st.session_state.final_status
        color = "#22c55e" if status == "HIRED" else "#ef4444" # Green or Red
        
        st.markdown(f"""
        <div style="background-color: #1e293b; padding: 40px; border-radius: 15px; text-align: center; border: 2px solid {color};">
            <h1 style="color: {color} !important; font-size: 60px; margin: 0;">{status}</h1>
            <h3 style="color: #cbd5e1;">Final Score: {avg_score:.1f}/10</h3>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📈 Score Breakdown")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CV/Resume", f"{st.session_state.scores['cv']}/10")
        col2.metric("Specialized", f"{st.session_state.scores['specialized']}/10")
        col3.metric("Attitude", f"{st.session_state.scores['attitude']}/10")
        col4.metric("Thinking", f"{st.session_state.scores['thinking']}/10")
        
        st.markdown("### 📝 AI Feedback & Suggestions")
        st.markdown(st.session_state.final_feedback)
        
        if st.button("Opportunities"):
            show_opportunities_popup()
