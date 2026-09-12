import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from collections import Counter

# 1. ตั้งค่าโครงสร้างหน้าเว็บ
st.set_page_config(
    page_title="Metal Nut Quality Control Platform",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. จัดการ State สำหรับนับจำนวนการตรวจและควบคุม Workflow
if 'total_scanned' not in st.session_state:
    st.session_state.total_scanned = 0
if 'pass_count' not in st.session_state:
    st.session_state.pass_count = 0
if 'fail_count' not in st.session_state:
    st.session_state.fail_count = 0
if 'qc_stage' not in st.session_state:
    st.session_state.qc_stage = 'capture'  # Stages: capture, confirm, result

# 3. CSS ปรับแต่งความสวยงาม ยกระดับ UI/UX
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Kanit', sans-serif;
    }

    .stApp {
        background-color: #FAF5F7 !important;
        color: #1F2937 !important;
    }
    
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] span {
        color: #374151 !important;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FDF2F4 !important;
        border-right: 2px solid #FBCFE8 !important;
    }
    
    .sidebar-header {
        color: #881337 !important;
        font-weight: 700;
        font-size: 20px;
        margin-bottom: 10px;
    }

    /* Step Card */
    .step-card {
        background-color: #FFFFFF;
        border-left: 4px solid #F43F5E;
        padding: 12px;
        margin-bottom: 10px;
        border-radius: 6px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
    }
    .step-title {
        font-weight: 600;
        color: #9F1239 !important;
    }

    /* Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFE4E6 100%);
        padding: 20px 25px;
        border-radius: 16px;
        border: 1px solid #FECDD3;
        box-shadow: 0 4px 15px rgba(225, 29, 72, 0.05);
        margin-bottom: 20px;
    }

    /* Focus Frame Guide */
    .focus-guide {
        border: 2px dashed #F43F5E;
        background-color: rgba(244, 63, 94, 0.03);
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        color: #9F1239;
        font-weight: 500;
        margin-bottom: 15px;
    }

    /* Status Cards */
    .status-pass {
        background-color: #ECFDF5;
        border: 2px solid #34D399;
        color: #065F46 !important;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
    }

    .status-fail {
        background-color: #FFF1F2;
        border: 2px solid #F87171;
        color: #991B1B !important;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
    }

    /* Metric Box */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #FFE4E6;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 4. โหลดโมเดล YOLO
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 5. แถบข้าง (Sidebar) - การตั้งค่าและสถิติ
with st.sidebar:
    st.markdown("<div class='sidebar-header'>⚙️ การตั้งค่าระบบ</div>", unsafe_allow_html=True)
    
    # ความไว AI Slider
    conf_threshold = st.slider(
        "🎯 ความไวการตรวจจับ AI (Confidence)", 
        min_value=0.2, 
        max_value=0.9, 
        value=0.5, 
        step=0.05,
        help="ปรับค่าความเชื่อมั่นของ AI ยิ่งค่าสูง AI จะสแกนเฉพาะจุดที่มั่นใจมากๆ เท่านั้น"
    )
    
    st.markdown("<hr style='border-color:#FBCFE8;'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-header'>📊 สรุปยอดการตรวจ (Shift Summary)</div>", unsafe_allow_html=True)
    
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("จำนวนที่ตรวจ", f"{st.session_state.total_scanned} ชิ้น")
    
    pass_rate = (st.session_state.pass_count / st.session_state.total_scanned * 100) if st.session_state.total_scanned > 0 else 0
    col_sb2.metric("อัตราผ่าน (Pass Rate)", f"{pass_rate:.1f}%")
    
    col_sb3, col_sb4 = st.columns(2)
    col_sb3.metric("🟢 PASS", f"{st.session_state.pass_count}")
    col_sb4.metric("🔴 FAIL", f"{st.session_state.fail_count}")

    if st.button("🗑️ รีเซ็ตสถิติกะนี้", use_container_width=True):
        st.session_state.total_scanned = 0
        st.session_state.pass_count = 0
        st.session_state.fail_count = 0
        st.rerun()

    st.markdown("<hr style='border-color:#FBCFE8;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class="step-card">
        <div class="step-title">📌 วิธีตรวจชิ้นงาน</div>
        1. วางน็อตให้อยู่ในระยะกรอบแนะนำ<br>
        2. กด <b>Take Photo</b> แล้วกด <b>ยืนยันส่งตรวจ</b><br>
        3. ตรวจสอบผล และกด <b>ตรวจสอบชิ้นถัดไป</b>
    </div>
    """, unsafe_allow_html=True)

# 6. Header หลัก
st.markdown("""
<div class="header-banner">
    <h1 style="color: #881337; margin:0; font-size:26px;">🔩 ระบบตรวจจับและตรวจสอบคุณภาพน็อตโลหะเรียลไทม์</h1>
    <p style="color: #9F1239; margin:5px 0 0 0; font-size:14px;">Automated AI Industrial Quality Inspection Platform</p>
</div>
""", unsafe_allow_html=True)

# 7. แบ่งเลย์เอาต์หลัก 2 คอลัมน์
col_cam, col_result = st.columns([1.1, 1], gap="large")

with col_cam:
    st.markdown("<h3 style='color: #881337;'>📸 1. บันทึกและจัดระยะภาพ (Camera Capture)</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="focus-guide">
        🎯 <b>คำแนะนำโฟกัส:</b> จัดวางน็อตให้อยู่กึ่งกลางกล้อง รักษาระยะห่าง 10-15 ซม. และหลีกเลี่ยงแสงสะท้อนจ้า
    </div>
    """, unsafe_allow_html=True)
    
    img_file_buffer = st.camera_input("", help="กดถ่ายภาพชิ้นงานน็อตโลหะ")
    
    if img_file_buffer is not None:
        st.image(img_file_buffer, caption="📷 ตัวอย่างภาพถ่ายเตรียมส่งตรวจ", use_container_width=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("✅ ยืนยันใช้รูปนี้ส่งตรวจ", type="primary", use_container_width=True):
                st.session_state.qc_stage = 'analyze'
        with col_btn2:
            if st.button("🔄 ถ่ายรูปใหม่", use_container_width=True):
                st.session_state.qc_stage = 'capture'
                st.rerun()

with col_result:
    st.markdown("<h3 style='color: #881337;'>📊 2. ผลการวิเคราะห์และตรวจสอบ (QC Analysis)</h3>", unsafe_allow_html=True)
    
    if img_file_buffer is not None and st.session_state.qc_stage == 'analyze':
        with st.spinner("🔍 AI กำลังประมวลผลวิเคราะห์จุดบกพร่อง..."):
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # AI Inference
            results = model.predict(source=cv2_img, conf=conf_threshold, verbose=False)
            res = results[0]
            total_defects = len(res.boxes)

            # อัปเดตสถิติ
            st.session_state.total_scanned += 1

            if total_defects == 0:
                st.session_state.pass_count += 1
                annotated_frame = cv2_img.copy()
                cv2.putText(annotated_frame, "QC: PASS (GOOD)", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                st.markdown('<div class="status-pass">🟢 สถานะชิ้นงาน: PASS (ผ่านเกณฑ์)</div>', unsafe_allow_html=True)
                st.success("✨ ชิ้นงานสมบูรณ์แบบ ไม่พบรอยแตกร้าวหรือรอยขีดข่วน")
            else:
                st.session_state.fail_count += 1
                annotated_frame = res.plot()
                cv2.putText(annotated_frame, f"QC: FAIL ({total_defects})", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                st.markdown(f'<div class="status-fail">🔴 สถานะชิ้นงาน: FAIL (พบตำหนิ {total_defects} จุด)</div>', unsafe_allow_html=True)
                
                class_ids = res.boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[i] for i in class_ids]
                counts = Counter(class_names)
                
                crack_cnt = counts.get('crack', 0)
                scratch_cnt = counts.get('scratch', 0)
                
                m1, m2 = st.columns(2)
                m1.metric("💥 รอยแตกร้าว (Crack)", f"{crack_cnt} จุด")
                m2.metric("⚡ รอยขีดข่วน (Scratch)", f"{scratch_cnt} จุด")

            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="ภาพผลการวิเคราะห์จาก AI", use_container_width=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            # ปุ่มตรวจสอบชิ้นถัดไป
            if st.button("⏭️ ตรวจสอบชิ้นถัดไป (Inspect Next Item)", type="primary", use_container_width=True):
                st.session_state.qc_stage = 'capture'
                st.rerun()

    else:
        st.info("👈 **ขั้นตอน:** ถ่ายภาพชิ้นงานทางฝั่งซ้าย -> กดปุ่ม '✅ ยืนยันใช้รูปนี้ส่งตรวจ' เพื่อเริ่มต้นวิเคราะห์ผล")
