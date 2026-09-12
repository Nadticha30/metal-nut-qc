import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from collections import Counter

# 1. ตั้งค่าโครงสร้างหน้าเว็บ
st.set_page_config(
    page_title="Metal Nut Quality Control",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ปรับแต่ง CSS บังคับสีโทนขาว-ชมพูพาสเทล ให้ตัวหนังสือน้ำตาลเข้ม/ชมพูเข้ม คมชัด อ่านง่าย
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Kanit', sans-serif;
    }

    /* พื้นหลังหลักสีขาวนวลอมชมพูพาสเทล */
    .stApp {
        background-color: #FAF5F7 !important;
        color: #1F2937 !important;
    }
    
    /* บังคับตัวหนังสือใน Markdown ทั้งหมดให้เป็นสีเข้ม */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] li,
    div[data-testid="stMarkdownContainer"] span {
        color: #374151 !important;
        font-size: 15px;
    }

    /* แถบด้านข้าง (Sidebar) ธีมชมพูโรสโกลด์อ่อน */
    [data-testid="stSidebar"] {
        background-color: #FDF2F4 !important;
        border-right: 2px solid #FBCFE8 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #881337 !important;
    }

    /* กล่องข้อความสเต็ปแนะนำใน Sidebar */
    .step-card {
        background-color: #FFFFFF;
        border-left: 4px solid #F43F5E;
        padding: 12px 15px;
        margin-bottom: 12px;
        border-radius: 4px 10px 10px 4px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    
    .step-title {
        font-weight: 600;
        color: #9F1239 !important;
        margin-bottom: 4px;
    }

    /* หัวข้อหลัก Main Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFE4E6 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #FECDD3;
        box-shadow: 0 4px 15px rgba(225, 29, 72, 0.05);
        margin-bottom: 25px;
    }

    /* ป้ายสถานะ PASS (สีเขียวคมชัด) */
    .status-pass {
        background-color: #ECFDF5;
        border: 2px solid #34D399;
        color: #065F46 !important;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    /* ป้ายสถานะ FAIL (สีแดงคมชัด) */
    .status-fail {
        background-color: #FFF1F2;
        border: 2px solid #F87171;
        color: #991B1B !important;
        padding: 16px;
        border-radius: 12px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    /* การ์ดรองรับข้อมูล */
    .info-box {
        background-color: #FFFFFF;
        border: 1px solid #FFE4E6;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. โหลดโมเดล YOLO
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 4. แถบแนะนำการใช้งานฝั่งซ้าย (Sidebar)
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>📋 คู่มือการใช้งาน</h2>", unsafe_allow_html=True)
    st.markdown("<hr style='margin-top:0; margin-bottom:15px; border-color:#FBCFE8;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-card">
        <div class="step-title">📸 สเต็ปที่ 1: จัดวางชิ้นงาน</div>
        <div>วางน็อตโลหะให้อยู่กึ่งกลางของระยะกล้อง</div>
    </div>
    
    <div class="step-card">
        <div class="step-title">🔘 สเต็ปที่ 2: ถ่ายภาพ</div>
        <div>กดปุ่ม <b>Take Photo</b> ทางฝั่งซ้ายเพื่อบันทึกภาพ</div>
    </div>
    
    <div class="step-card">
        <div class="step-title">⚡ สเต็ปที่ 3: รอประมวลผล</div>
        <div>ระบบ AI จะวิเคราะห์จุดบกพร่องให้อัตโนมัติ</div>
    </div>
    
    <div class="step-card">
        <div class="step-title">📊 สเต็ปที่ 4: ตรวจสอบผล</div>
        <div>ดูผลสถานะ <b>PASS/FAIL</b> และตำแหน่งตำหนิทางฝั่งขวา</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    st.info("💡 **ข้อแนะนำ:** ควรจัดวางชิ้นงานบนพื้นหลังสีเข้ม และมีแสงสว่างเพียงพอเพื่อความแม่นยำสูงสุด")
    st.caption("🤖 **AI Model Version:** YOLO11 V2 Final")

# 5. ส่วนหัวหน้าเว็บ (Header Banner)
st.markdown("""
<div class="header-banner">
    <h1 style="color: #881337; margin:0; font-size:26px;">🔩 ระบบตรวจจับและตรวจสอบคุณภาพน็อตโลหะเรียลไทม์</h1>
    <p style="color: #9F1239; margin:5px 0 0 0; font-size:14px;">Automated Industrial Visual Quality Inspection System</p>
</div>
""", unsafe_allow_html=True)

# 6. แบ่งเลย์เอาต์ 2 คอลัมน์
col_cam, col_result = st.columns([1.1, 1], gap="large")

with col_cam:
    st.markdown("<h3 style='color: #881337;'>📸 1. ถ่ายภาพชิ้นงาน (Camera Input)</h3>", unsafe_allow_html=True)
    st.caption("จัดตำแหน่งน็อตแล้วกดถ่ายภาพด้านล่างเพื่อส่งตรวจ QC")
    
    img_file_buffer = st.camera_input("", help="กดถ่ายภาพเพื่อส่งวิเคราะห์ QC")

with col_result:
    st.markdown("<h3 style='color: #881337;'>📊 2. ผลการวิเคราะห์ (QC Results)</h3>", unsafe_allow_html=True)
    
    if img_file_buffer is not None:
        with st.spinner("🔍 AI กำลังประมวลผลและตรวจหารอยตำหนิ..."):
            # แปลงไฟล์ภาพ
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # AI ทำนายผล
            results = model.predict(source=cv2_img, conf=0.5, verbose=False)
            res = results[0]
            total_defects = len(res.boxes)

            # แสดงสถานะ PASS / FAIL
            if total_defects == 0:
                annotated_frame = cv2_img.copy()
                cv2.putText(annotated_frame, "QC: PASS (GOOD)", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                st.markdown('<div class="status-pass">🟢 สถานะชิ้นงาน: PASS (สมบูรณ์แบบ)</div>', unsafe_allow_html=True)
                
                st.markdown("""
                <div class="info-box">
                    <h4 style="color:#065F46; margin:0;">✨ ไม่พบจุดบกพร่องใดๆ</h4>
                    <p style="margin-top:5px; color:#4B5563;">ชิ้นงานผ่านเกณฑ์มาตรฐาน สามารถนำไปใช้งานหรือจัดส่งตามปกติ</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                annotated_frame = res.plot()
                cv2.putText(annotated_frame, f"QC: FAIL ({total_defects})", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                st.markdown(f'<div class="status-fail">🔴 สถานะชิ้นงาน: FAIL (พบตำหนิ {total_defects} จุด)</div>', unsafe_allow_html=True)
                
                # นับประเภทตำหนิ
                class_ids = res.boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[i] for i in class_ids]
                counts = Counter(class_names)
                
                crack_cnt = counts.get('crack', 0)
                scratch_cnt = counts.get('scratch', 0)
                
                m1, m2 = st.columns(2)
                m1.metric(label="💥 รอยแตกร้าว (Crack)", value=f"{crack_cnt} จุด")
                m2.metric(label="⚡ รอยขีดข่วน (Scratch)", value=f"{scratch_cnt} จุด")

            st.write("")
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="ภาพบันทึกผลการตรวจจับจาก AI", use_container_width=True)
            
    else:
        st.markdown("""
        <div class="info-box" style="text-align: center; padding: 40px 20px;">
            <p style="font-size: 40px; margin:0;">👈</p>
            <h4 style="color: #881337; margin-top: 10px;">พร้อมทำการตรวจ QC</h4>
            <p style="color: #6B7280;">กรุณากดปุ่ม <b>Take Photo</b> ที่กล่องถ่ายรูปทางฝั่งซ้าย เพื่อเริ่มการวิเคราะห์ภาพถ่าย</p>
        </div>
        """, unsafe_allow_html=True)
