import streamlit as st
import google.generativeai as genai
import time
import PyPDF2
import docx2txt
import os

# ==============================================================================
# 🔑 CẤU HÌNH API KEY (TỰ ĐỘNG XỬ LÝ)
# ==============================================================================
# Logic: 
# 1. Ưu tiên lấy từ Secrets (khi deploy lên Streamlit Cloud).
# 2. Nếu không có Secrets, dùng Key cứng bạn điền (khi chạy máy local).
# ==============================================================================

try:
    # Thử lấy key từ hệ thống bảo mật (khi deploy)
    API_KEY = st.secrets["GOOGLE_API_KEY"]
except:
    # Nếu không có, dùng key cứng của bạn (Thay key của bạn vào dấu ngoặc kép bên dưới)
    # LƯU Ý: Key bạn gửi trong tin nhắn cũ đã bị lộ, tôi để trống để bảo mật. Hãy điền lại.
    API_KEY = "" 

# ==============================================================================

# --- 1. CẤU HÌNH TRANG WEB & CSS ---
st.set_page_config(
    page_title="AI Recruiter Pro",
    page_icon="🕴️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS: Tối ưu hiển thị (Dark Mode Report Card + Clean UI)
st.markdown("""
<style>
    /* Ẩn Header/Footer mặc định của Streamlit */
    header, footer {visibility: hidden;}
    .main { background-color: #f8f9fa; }
    
    /* Bong bóng chat User */
    div[data-testid="user-message"] {
        background-color: #e3f2fd; 
        border-radius: 20px 20px 5px 20px; 
        padding: 15px; 
        color: #0d47a1; 
        border: 1px solid #bbdefb;
    }
    
    /* Bong bóng chat AI */
    div[data-testid="assistant-message"] {
        background-color: #ffffff; 
        border-radius: 20px 20px 20px 5px; 
        padding: 15px; 
        color: #2c3e50; 
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    /* REPORT CARD - GIAO DIỆN DARK MODE CAO CẤP */
    .report-card {
        background-color: #1e1e1e !important; /* Nền đen dịu */
        padding: 30px; 
        border-radius: 15px; 
        border: 1px solid #333;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        margin-top: 20px;
        color: #e0e0e0 !important; /* Chữ trắng xám */
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Tiêu đề chính */
    .report-card h1 {
        color: #64b5f6 !important; /* Xanh dương sáng */
        border-bottom: 2px solid #64b5f6;
        padding-bottom: 10px;
        margin-top: 20px;
        font-weight: 700;
        text-transform: uppercase;
    }
    
    /* Tiêu đề mục con */
    .report-card h2 {
        color: #ffd54f !important; /* Vàng kim loại */
        margin-top: 25px;
        font-weight: 600;
        border-left: 4px solid #ffd54f;
        padding-left: 10px;
    }
    
    .report-card h3 {
        color: #81c784 !important; /* Xanh lá */
        margin-top: 15px;
    }
    
    .report-card strong {
        color: #ffb74d !important; /* Cam sáng */
    }
    
    .report-card ul, .report-card li {
        color: #e0e0e0 !important;
        line-height: 1.6;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOGIC XỬ LÝ (BACKEND) ---

def extract_text_from_cv(uploaded_file):
    """Đọc file PDF/DOCX"""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(uploaded_file)
            return " ".join([page.extract_text() for page in reader.pages])
        elif "word" in uploaded_file.type:
            return docx2txt.process(uploaded_file)
        else:
            return uploaded_file.read().decode("utf-8")
    except Exception: return ""

def get_interviewer_prompt(job_title, job_desc, exp_level, cv_text, is_demo):
    """
    Prompt thông minh: Chia giai đoạn phỏng vấn
    """
    cv_context = f"\nTHÔNG TIN CV ỨNG VIÊN:\n{cv_text[:3000]}\n(Hãy dùng thông tin này để 'bẻ' ứng viên)" if cv_text else ""
    
    if is_demo:
        # LOGIC DEMO 3 CÂU (SMART DEMO)
        return f"""
        [CHẾ ĐỘ DEMO - RÚT GỌN 3 VÒNG]
        ROLE: CTO khó tính tuyển vị trí {job_title} ({exp_level}).
        JD: {job_desc}
        {cv_context}
        
        NHIỆM VỤ: Bạn phải thực hiện phỏng vấn ĐÚNG 3 CÂU (3 Round) để đánh giá nhanh nhưng toàn diện.
        
        QUY TRÌNH BẮT BUỘC:
        1. Bỏ qua chào hỏi rườm rà. Vào thẳng vấn đề.
        2. ROUND 1 (Tech Core): Hỏi 1 câu kỹ thuật chuyên sâu nhất liên quan đến JD hoặc dự án trong CV.
        3. ROUND 2 (Mindset): Sau khi ứng viên trả lời câu 1, hãy hỏi tiếp 1 câu về Tư duy giải quyết vấn đề (Problem Solving) hoặc System Design.
        4. ROUND 3 (Culture): Sau câu 2, hỏi 1 câu về Thái độ/Làm việc nhóm (Behavioral).
        5. KẾT THÚC: Sau khi ứng viên trả lời câu 3, nói "Cảm ơn, buổi phỏng vấn kết thúc" và không hỏi thêm.
        
        LƯU Ý: Đừng hỏi 3 câu cùng lúc. Hỏi từng câu một. Chờ ứng viên trả lời rồi mới hỏi câu tiếp theo.
        """
    else:
        # LOGIC FULL (BÌNH THƯỜNG)
        return f"""
        ROLE: Nhà tuyển dụng chuyên nghiệp. Vị trí: {job_title} ({exp_level}).
        JD: {job_desc}
        {cv_context}
        
        QUY TRÌNH PHỎNG VẤN CHUẨN:
        1. Màn chào hỏi & Giới thiệu bản thân.
        2. Khai thác kinh nghiệm trong CV (Deep dive vào các dự án cũ).
        3. Kiểm tra kiến thức nền tảng (Foundation).
        4. Kiểm tra kiến thức chuyên sâu/Coding (Advanced).
        5. Câu hỏi tình huống (Soft skills).
        6. Tổng kết.
        
        PHONG CÁCH:
        - Hỏi từng câu một.
        - Nếu ứng viên trả lời sai/thiếu, hãy challenge lại (Vd: "Tại sao bạn không dùng X thay vì Y?").
        - Tương tác tự nhiên như người thật.
        """

def get_evaluation_prompt(job_role, exp_level):
    """
    Prompt đánh giá thông minh: Xử lý cả trường hợp kết thúc sớm.
    """
    return f"""
    Hãy đóng vai Hội đồng tuyển dụng, phân tích lịch sử chat và tạo BÁO CÁO TUYỂN DỤNG (Markdown) cho vị trí {job_role} ({exp_level}).
    
    KIỂM TRA TRẠNG THÁI:
    - Nếu ứng viên trả lời đủ các vòng -> Đánh giá toàn diện.
    - Nếu ứng viên DỪNG SỚM (chưa trả lời hết) -> Chỉ chấm điểm phần đã làm. Phần chưa làm thì đưa ra "Gợi ý/Đáp án mẫu" để họ học hỏi.
    
    CẤU TRÚC BÁO CÁO (BẮT BUỘC - DARK MODE):
    
    # 📑 BẢNG ĐIỂM CHI TIẾT
    
    ## 1. 🎯 TỔNG QUAN
    * **Quyết định:** [PASS / FAIL / CÂN NHẮC]
    * **Điểm số:** .../10
    * **Nhận xét chung:** (Tóm tắt 2 dòng về ấn tượng)
    
    ## 2. 🔍 ĐÁNH GIÁ CHI TIẾT
    *Phân tích kỹ các câu trả lời của ứng viên:*
    * **Chuyên môn (Hard Skills):** ...
    * **Tư duy (Problem Solving):** ...
    * **Thái độ (Soft Skills):** ...
    *(Ghi chú rõ điểm mạnh/yếu)*
    
    ## 3. 💡 KIẾN THỨC BỔ SUNG
    *Dành cho các câu hỏi ứng viên trả lời sai HOẶC chưa kịp hỏi do dừng sớm:*
    * **Vấn đề:** ...
    * **Giải pháp chuẩn:** ...
    * **Từ khóa cần học:** (Ví dụ: SOLID, ACID, CAP Theorem...)
    
    ## 4. 🚀 LỜI KHUYÊN PHÁT TRIỂN
    * **Tips cải thiện:** ...
    * **Tài liệu gợi ý:** ...
    """

# --- 3. SIDEBAR (CONTROL PANEL) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/942/942751.png", width=60)
    st.title("🎛️ Control Panel")
    
    # Hiển thị trạng thái API
    if API_KEY:
        st.success("✅ Đã kết nối API")
    else:
        st.error("⚠️ Chưa có API Key")
        st.info("Vui lòng điền Key vào file code hoặc cấu hình Secrets trên Cloud.")
    
    st.markdown("---")
    
    # 1. Upload CV
    st.subheader("1. Hồ sơ Ứng viên (CV)")
    uploaded_file = st.file_uploader("Upload CV (PDF/Word)", type=['pdf', 'docx', 'txt'])
    
    # Nút Phân tích CV nhanh
    if uploaded_file and API_KEY:
        if st.button("🔍 Phân tích CV (AI Scan)", use_container_width=True):
            with st.spinner("Đang đọc CV..."):
                try:
                    genai.configure(api_key=API_KEY)
                    cv_raw = extract_text_from_cv(uploaded_file)
                    st.session_state.cv_text = cv_raw
                    # Dùng model xịn để scan CV
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    summary = model.generate_content(f"Đọc CV này và trích xuất 3 kỹ năng chính ngắn gọn: {cv_raw[:2000]}")
                    st.session_state.cv_summary = summary.text
                except Exception as e:
                    st.error(f"Lỗi phân tích CV: {e}")
                
    if "cv_summary" in st.session_state:
        st.success("✅ Đã đọc xong CV")
        st.info(f"**Kỹ năng tìm thấy:**\n{st.session_state.cv_summary}")

    st.markdown("---")

    # 2. Cấu hình Phỏng vấn
    st.subheader("2. Thiết lập Vị trí")
    
    # Toggle Demo Mode
    is_demo = st.toggle("⚡ Chế độ Demo (3 Câu hỏi)", value=True, help="Hỏi 3 câu trọng tâm (Tech -> Mindset -> Culture) rồi kết thúc.")
    
    job_role = st.selectbox("Vị trí ứng tuyển", [
        "Frontend Developer (ReactJS)", 
        "Backend Developer (NodeJS/Go)", 
        "Fullstack Developer",
        "Data Scientist / AI Engineer",
        "DevOps Engineer",
        "Business Analyst (BA)"
    ])
    
    # Dùng Selectbox thay cho Slider để không bị chồng chữ
    exp_level = st.selectbox("Mức độ kinh nghiệm", [
        "Intern (Thực tập sinh)",
        "Fresher (Mới ra trường)",
        "Junior (1-2 năm)",
        "Mid-Level (3-4 năm)",
        "Senior (5+ năm)",
        "Lead / Manager"
    ], index=2)
    
    # JD Tự động điền (ngắn gọn)
    default_jd = "- Nắm vững DSA, OOP.\n- Kỹ năng giải quyết vấn đề."
    if "Frontend" in job_role: default_jd = "- ReactJS, Redux, NextJS.\n- Tối ưu Performance, SEO.\n- Responsive Design."
    if "Backend" in job_role: default_jd = "- Microservices, System Design.\n- Database (SQL/NoSQL), Caching.\n- Cloud (AWS/Docker)."
    if "Data" in job_role: default_jd = "- Python, Pandas, SQL.\n- Machine Learning Models.\n- Data Visualization."
    
    job_desc = st.text_area("Yêu cầu công việc (JD)", value=default_jd, height=100)
    
    st.markdown("---")
    
    # Nút Start
    col1, col2 = st.columns(2)
    start_btn = col1.button("▶️ BẮT ĐẦU", type="primary", use_container_width=True)
    reset_btn = col2.button("🔄 RESET", use_container_width=True)

# --- 4. LOGIC CHÍNH (MAIN APP) ---

# Init Session
if "messages" not in st.session_state: st.session_state.messages = []
if "chat" not in st.session_state: st.session_state.chat = None
if "active" not in st.session_state: st.session_state.active = False
if "cv_text" not in st.session_state: st.session_state.cv_text = ""

# Reset
if reset_btn:
    st.session_state.messages = []
    st.session_state.chat = None
    st.session_state.active = False
    st.session_state.cv_text = ""
    if "cv_summary" in st.session_state: del st.session_state.cv_summary
    st.rerun()

# Start
if start_btn:
    if not API_KEY:
        st.error("⚠️ Chưa nhập API Key! Vui lòng kiểm tra lại file code hoặc Secrets.")
    else:
        genai.configure(api_key=API_KEY)
        
        # Nếu chưa upload CV thì lấy text rỗng
        if uploaded_file and not st.session_state.cv_text:
             st.session_state.cv_text = extract_text_from_cv(uploaded_file)
        
        # Tạo Prompt
        sys_prompt = get_interviewer_prompt(job_role, job_desc, exp_level, st.session_state.cv_text, is_demo)
        
        # Init Model - DÙNG GEMINI 2.5 FLASH
        try:
            model = genai.GenerativeModel('gemini-2.5-flash', system_instruction=sys_prompt)
            st.session_state.chat = model.start_chat(history=[])
            st.session_state.active = True
            st.session_state.messages = []
            
            # Câu mở đầu
            if is_demo:
                welcome = f"🔥 **[DEMO 3 ROUNDS - GEMINI 2.5]** Chào bạn. Tôi là AI CTO. Chúng ta sẽ đi qua 3 câu hỏi trọng tâm: **Tech Core ➔ Mindset ➔ Culture**. \n\nTôi đã đọc CV của bạn. Hãy sẵn sàng cho câu hỏi đầu tiên (Round 1) ngay sau đây."
            else:
                welcome = f"Chào bạn, tôi là trợ lý tuyển dụng ảo. Rất vui được phỏng vấn bạn cho vị trí **{job_role}**. Chúng ta sẽ bắt đầu nhé."
                
            st.session_state.messages.append({"role": "assistant", "content": welcome})
            
            # Trigger câu hỏi đầu tiên
            if is_demo:
                with st.spinner("Gemini 2.5 đang nghiên cứu CV..."):
                    response = st.session_state.chat.send_message("Bắt đầu Round 1 ngay.")
                    st.session_state.messages.append({"role": "assistant", "content": response.text})
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Lỗi khởi tạo Gemini 2.5: {e}")
            if "429" in str(e):
                st.warning("Gợi ý: Nếu model 2.5 vẫn quá tải, hãy thử đổi code về 'gemini-1.5-flash' để ổn định tuyệt đối.")

# --- 5. GIAO DIỆN CHAT ---

st.title("🕴️ AI Tech Interviewer Pro (v2.6)")

if st.session_state.active:
    # Hiển thị thông báo chế độ
    mode_text = "⚡ CHẾ ĐỘ DEMO (3 CÂU)" if is_demo else "🐢 CHẾ ĐỘ PHỎNG VẤN ĐẦY ĐỦ"
    st.caption(f"{mode_text} | Vị trí: {job_role} | Level: {exp_level}")
    
    # Render Chat
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "🧑‍💻"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            
    # Input
    if user_input := st.chat_input("Trả lời phỏng vấn..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(user_input)
            
        with st.chat_message("assistant", avatar="🤖"):
            placeholder = st.empty()
            placeholder.markdown("⏳ *Gemini 2.5 đang suy nghĩ...*")
            try:
                response = st.session_state.chat.send_message(user_input)
                placeholder.markdown(response.text)
                st.session_state.messages.append({"role": "assistant", "content": response.text})
            except Exception as e:
                placeholder.empty()
                if "429" in str(e):
                    st.error("⚠️ Server quá tải! Vui lòng đợi 10s rồi thử lại.")
                else:
                    st.error(f"Lỗi: {e}")

    # Nút Kết thúc
    st.markdown("---")
    # Thay đổi nhãn nút tùy theo trạng thái
    finish_label = "✋ DỪNG SỚM & NHẬN GỢI Ý (Nếu chưa xong)" if len(st.session_state.messages) < 6 else "🏁 KẾT THÚC & CHẤM ĐIỂM"
    
    if st.button(finish_label, type="primary", use_container_width=True):
        if st.session_state.chat:
            with st.spinner("Đang phân tích và tổng hợp kiến thức..."):
                try:
                    # Truyền thêm tham số ngữ cảnh vào hàm tạo prompt đánh giá
                    eval_prompt = get_evaluation_prompt(job_role, exp_level)
                    final = st.session_state.chat.send_message(eval_prompt)
                    st.markdown(f"<div class='report-card'>{final.text}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error("Lỗi khi tạo báo cáo. Vui lòng thử lại sau vài giây.")

else:
    # Màn hình chờ
    st.info("👈 Vui lòng nhập API Key, Upload CV (nếu có) và nhấn START.")
    st.markdown("""
    ### 🚀 Tính năng mới v2.6:
    1.  **Chấm điểm thông minh:** Dừng sớm vẫn được chấm điểm phần đã làm.
    2.  **Kiến thức bổ sung:** AI sẽ tự động gợi ý đáp án cho các phần bạn chưa kịp trả lời.
    3.  **Giao diện:** Đã sửa lỗi hiển thị báo cáo (màu chữ tối trên nền trắng).
    """)