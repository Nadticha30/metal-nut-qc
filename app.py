import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from collections import Counter

# 1. ตั้งค่าโครงสร้างหน้าเว็บ
st.set_page_config(
    page_title="AI Metal Nut QC Inspection",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. ปรับแต่งธีมสีขาว-ชมพูพาสเทล (Soft Pink) และตกแต่ง CSS ให้เป็นทางการ
custom_css = """
<style>
    /* ธีมพื้นหลังหลักสีขาว-ชมพูอ่อน */
    .stApp {
        background-color: #FAFAFC;
    }
    
    /* ตกแต่งแถบด้านข้าง (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #FFF0F5;
        border-right: 1px solid #FFE4E1;
    }

    /* ตกแต่งการ์ดแสดงผล */
    .metric-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(244, 114, 182, 0.08);
        border: 1px solid #FCE7F3;
        margin-bottom: 15px;
    }

    /* หัวข้อและตัวหนังสือ */
    h1, h2, h3 {
        color: #831843 !important;
        font-family: 'Kanit', sans-serif;
    }

    /* ปรับป้ายสถานะ PASS / FAIL */
    .status-pass {
        background-color: #ECFDF5;
        color: #047857;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 20px;
        border: 1px solid #A7F3D0;
        text-align: center;
    }

    .status-fail {
        background-color: #FFF1F2;
        color: #BE123C;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 20px;
        border: 1px solid #FECDD3;
        text-align: center;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# 3. โหลดโมเดล YOLO
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 4. แถบแนะนำการใช้งาน (Sidebar)
with st.sidebar:
    st.image("https://img.icons8.com/isometric-folders/100/inspection.png", width=70)
    st.title("📌 คู่มือการใช้งาน")
    st.markdown("""
    **ขั้นตอนการตรวจจับคุณภาพ:**
    1. **สเต็ป 1:** วางน็อตให้อยู่ในระยะกล้อง
    2. **สเต็ป 2:** กดปุ่ม **Take Photo** เพื่อถ่ายภาพ
    3. **สเต็ป 3:** รอ AI วิเคราะห์ผลสักครู่
    4. **สเต็ป 4:** ตรวจสอบผล `PASS/FAIL` ที่ฝั่งขวา
    
    ---
    💡 *คำแนะนำ: แนะนำให้วางน็อตบนพื้นหลังสีเข้มเพื่อความแม่นยำสูงสุด*
    """)
    st.info("🤖 **Model Version:** YOLO11 V2 Final")

# 5. ส่วนหัวของหน้าเว็บ Main Dashboard
st.title("🔩 ระบบตรวจจับและตรวจสอบคุณภาพน็อตโลหะเรียลไทม์")
st.caption("Automated Visual Inspection & Quality Control Dashboard")
st.markdown("---")

# จัดสรรพื้นที่ 2 คอลัมน์หลัก
col_cam, col_result = st.columns([1.1, 1], gap="large")

with col_cam:
    st.subheader("📷 1. ถ่ายภาพชิ้นงาน (Camera Input)")
    st.caption("กดปุ่มถ่ายภาพด้านล่างเมื่อจัดตำแหน่งชิ้นงานเรียบร้อยแล้ว")
    
    img_file_buffer = st.camera_input("", help="กดถ่ายภาพเพื่อส่งวิเคราะห์ QC")

with col_result:
    st.subheader("📊 2. ผลการวิเคราะห์คุณภาพ (QC Results)")
    
    if img_file_buffer is not None:
        # แสดงสถานะกำลังประมวลผล
        with st.spinner("🔍 AI กำลังประมวลผลและตรวจหารอยตำหนิ..."):
            # แปลงไฟล์ภาพ
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            # AI ทำนายผล
            results = model.predict(source=cv2_img, conf=0.5, verbose=False)
            res = results[0]
            total_defects = len(res.boxes)

            # ประมวลผลสถานะ PASS / FAIL
            if total_defects == 0:
                annotated_frame = cv2_img.copy()
                cv2.putText(annotated_frame, "QC: PASS (GOOD)", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                
                st.markdown('<div class="status-pass">🟢 สถานะ: PASS (ชิ้นงานคุณภาพสมบูรณ์)</div>', unsafe_allow_html=True)
                st.write("")
                
                # แสดงการ์ดสรุปผล
                with st.container():
                    st.markdown("""
                    <div class="metric-card">
                        <h4 style="color:#047857; margin:0;">✨ ไม่พบจุดบกพร่อง</h4>
                        <p style="color:#4B5563; margin-top:5px;">ชิ้นงานผ่านเกณฑ์มาตรฐาน QC สามารถนำไปใช้านได้ปกติ</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                annotated_frame = res.plot()
                cv2.putText(annotated_frame, f"QC: FAIL ({total_defects})", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                
                st.markdown(f'<div class="status-fail">🔴 สถานะ: FAIL (พบตำหนิ {total_defects} จุด)</div>', unsafe_allow_html=True)
                st.write("")
                
                # นับประเภทตำหนิ
                class_ids = res.boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[i] for i in class_ids]
                counts = Counter(class_names)
                
                crack_cnt = counts.get('crack', 0)
                scratch_cnt = counts.get('scratch', 0)
                
                # แสดงสรุปยอดตำหนิด้วย Metrics
                m1, m2 = st.columns(2)
                m1.metric(label="💥 รอยแตกร้าว (Crack)", value=f"{crack_cnt} จุด")
                m2.metric(label="⚡ รอยขีดข่วน (Scratch)", value=f"{scratch_cnt} จุด")

            # แสดงภาพผลลัพธ์การตรวจจับ
            st.write("")
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="ภาพบันทึกผลการตรวจจับจาก AI", use_container_width=True)
            
    else:
        # หน้าจอเริ่มต้นเมื่อยังไม่ได้ถ่ายภาพ
        st.info("👈 กรุณากดปุ่ม **Take Photo** ทางฝั่งซ้ายเพื่อเริ่มกระบวนการตรวจ QC")