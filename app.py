import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO
from collections import Counter

# 1. ตั้งค่าโครงสร้างหน้าเว็บ
st.set_page_config(
    page_title="M.A.T.R.I.X. Nut - QC Platform",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. จัดการ State สำหรับระบบ Workflow, Reset กล้อง และ Auto-Scroll
if 'total_scanned' not in st.session_state:
    st.session_state.total_scanned = 0
if 'pass_count' not in st.session_state:
    st.session_state.pass_count = 0
if 'fail_count' not in st.session_state:
    st.session_state.fail_count = 0
if 'qc_stage' not in st.session_state:
    st.session_state.qc_stage = 'capture'
if 'camera_key' not in st.session_state:
    st.session_state.camera_key = 0
if 'should_scroll' not in st.session_state:
    st.session_state.should_scroll = False

# ระบบ Scroll หน้าจอกลับขึ้นด้านบน (สำหรับมือถือ)
if st.session_state.should_scroll:
    st.session_state.should_scroll = False
    components.html(
        """
        <script>
            window.parent.scrollTo({top: 0, behavior: 'smooth'});
        </script>
        """,
        height=0
    )

# 3. CSS ปรับแต่งสี ความคมชัด และดีไซน์ปุ่มกด
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
        background-color: #FFFFFF !important;
        border-left: 4px solid #F43F5E !important;
        padding: 14px !important;
        margin-bottom: 12px !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05) !important;
        color: #1F2937 !important;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    .step-card p, .step-card span, .step-card div {
        color: #1F2937 !important;
    }

    .step-title {
        font-weight: 700 !important;
        color: #9F1239 !important;
        font-size: 16px !important;
        margin-bottom: 8px !important;
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
        background-color: rgba(244, 63, 94, 0.04);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        color: #9F1239 !important;
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

    /* ดีไซน์ปุ่มกด */
    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 10px 16px !important;
        transition: all 0.2s ease-in-out !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #E11D48 !important;
        color: #FFFFFF !important;
        border: 1px solid #BE123C !important;
        box-shadow: 0 3px 6px rgba(225, 29, 72, 0.25) !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #BE123C !important;
        border-color: #9F1239 !important;
        color: #FFFFFF !important;
    }

    div.stButton > button[kind="secondary"], div.stButton > button:not([kind="primary"]) {
        background-color: #FFFFFF !important;
        color: #1F2937 !important;
        border: 1.5px solid #D1D5DB !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04) !important;
    }

    div.stButton > button[kind="secondary"]:hover, div.stButton > button:not([kind="primary"]):hover {
        background-color: #F3F4F6 !important;
        border-color: #9CA3AF !important;
        color: #111827 !important;
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
    
    conf_threshold = st.slider(
        "🎯 ความไวการตรวจจับ AI (Confidence)", 
        min_value=0.2, 
        max_value=0.9, 
        value=0.5, 
        step=0.05
    )
    
    st.markdown("<hr style='border-color:#FBCFE8;'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-header'>📊 สรุปยอดการตรวจ (Shift Summary)</div>", unsafe_allow_html=True)
    
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("จำนวนที่ตรวจ", f"{st.session_state.total_scanned} ชิ้น")
    
    pass_rate = (st.session_state.pass_count / st.session_state.total_scanned * 100) if st.session_state.total_scanned > 0 else 0
    col_sb2.metric("อัตราผ่าน", f"{pass_rate:.1f}%")
    
    col_sb3, col_sb4 = st.columns(2)
    col_sb3.metric("🟢 PASS", f"{st.session_state.pass_count}")
    col_sb4.metric("🔴 FAIL", f"{st.session_state.fail_count}")

    if st.button("🗑️ รีเซ็ตสถิติ", use_container_width=True):
        st.session_state.total_scanned = 0
        st.session_state.pass_count = 0
        st.session_state.fail_count = 0
        st.rerun()

    st.markdown("<hr style='border-color:#FBCFE8;'>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="step-card">
        <div class="step-title">📌 วิธีตรวจชิ้นงาน</div>
        <p style="margin: 0 0 6px 0;">1. วางน็อตให้อยู่ในระยะกรอบแนะนำ</p>
        <p style="margin: 0 0 6px 0;">2. กด <b>Take Photo</b> แล้วกด <b>ยืนยันส่งตรวจ</b></p>
        <p style="margin: 0;">3. ตรวจสอบผล และกด <b>ตรวจสอบชิ้นถัดไป</b></p>
    </div>
    """, unsafe_allow_html=True)

# 6. Header หลัก (ปรับแก้เป็นชื่อใหม่ M.A.T.R.I.X. Nut เรียบร้อย)
st.markdown("""
<div class="header-banner">
    <h1 style="color: #881337; margin:0; font-size:30px; font-weight: 700;">🔩 M.A.T.R.I.X. Nut</h1>
    <p style="color: #9F1239; margin:5px 0 0 0; font-size:15px; font-weight: 500;">Metal Automated Testing & Real-time Inspection X System</p>
</div>
""", unsafe_allow_html=True)

# 7. แบ่งเลย์เอาต์หลัก 2 คอลัมน์
col_cam, col_result = st.columns([1.1, 1], gap="large")

with col_cam:
    st.markdown("<h3 style='color: #881337;'>📸 1. บันทึกและจัดระยะภาพ (Camera Capture)</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="focus-guide">
        🎯 <b>คำแนะนำโฟกัส:</b> จัดวางน็อตให้อยู่กึ่งกลางกล้อง รักษาระยะห่าง 10-15 ซม. และหลีกเลี่ยงแสงสะท้อน
    </div>
    """, unsafe_allow_html=True)
    
    img_file_buffer = st.camera_input("", key=f"cam_input_{st.session_state.camera_key}", help="กดถ่ายภาพชิ้นงานน็อตโลหะ")
    
    if img_file_buffer is not None:
        st.image(img_file_buffer, caption="📷 ตัวอย่างภาพถ่ายเตรียมส่งตรวจ", use_container_width=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("✅ ยืนยันใช้รูปนี้ส่งตรวจ", type="primary", use_container_width=True):
                st.session_state.qc_stage = 'analyze'
                st.rerun()
        with col_btn2:
            if st.button("🔄 ถ่ายรูปใหม่", use_container_width=True):
                st.session_state.camera_key += 1
                st.session_state.qc_stage = 'capture'
                st.session_state.should_scroll = True
                st.rerun()

with col_result:
    st.markdown("<h3 style='color: #881337;'>📊 2. ผลการวิเคราะห์และตรวจสอบ (QC Analysis)</h3>", unsafe_allow_html=True)
    
    if img_file_buffer is not None and st.session_state.qc_stage == 'analyze':
        with st.spinner("🔍 AI กำลังประมวลผลวิเคราะห์จุดบกพร่อง..."):
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            results = model.predict(source=cv2_img, conf=conf_threshold, verbose=False)
            res = results[0]
            total_defects = len(res.boxes)

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

            # แสดงรูปภาพวิเคราะห์
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, caption="ภาพผลการวิเคราะห์จาก AI", use_container_width=True)
            
            # 📝 กล่องสรุปรายละเอียดผลการตรวจใต้รูปภาพ
            st.markdown("""
            <div style='background-color: #FFFFFF; border: 1px solid #FECDD3; border-radius: 12px; padding: 16px; margin-top: 12px; margin-bottom: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.03);'>
                <h4 style='color: #881337; margin: 0 0 8px 0; font-size: 16px;'>📋 สรุปรายละเอียดผลการตรวจ (Detailed Result)</h4>
            """, unsafe_allow_html=True)
            
            if total_defects == 0:
                st.markdown("""
                <p style='color: #065F46; font-size: 15px; margin: 0; line-height: 1.6;'>
                    <b>ผลการประเมิน:</b> ✅ <span style='background-color:#E6F4EA; padding:2px 8px; border-radius:4px;'><b>ชิ้นงานดีเยี่ยม (PASS / GOOD)</b></span><br>
                    <b>คำอธิบาย:</b> ตรวจสอบผิวโลหะไม่พบรอยแตกร้าว (Crack) หรือรอยขีดข่วน (Scratch) ชิ้นงานได้มาตรฐาน สามารถส่งต่อกระบวนการถัดไปได้ทันที
                </p>
                """, unsafe_allow_html=True)
            else:
                defect_items = []
                for cls_name, cnt in counts.items():
                    if cls_name.lower() == 'crack':
                        defect_items.append(f"💥 <b>รอยแตกร้าว (Crack):</b> {cnt} จุด")
                    elif cls_name.lower() == 'scratch':
                        defect_items.append(f"⚡ <b>รอยขีดข่วน (Scratch):</b> {cnt} จุด")
                    else:
                        defect_items.append(f"⚠️ <b>{cls_name}:</b> {cnt} จุด")
                
                defect_str = "<br>• ".join(defect_items)
                
                st.markdown(f"""
                <p style='color: #991B1B; font-size: 15px; margin: 0; line-height: 1.6;'>
                    <b>ผลการประเมิน:</b> ❌ <span style='background-color:#FCE8E6; padding:2px 8px; border-radius:4px;'><b>ชิ้นงานชำรุด (FAIL / DEFECTIVE)</b></span><br>
                    <b>รายละเอียดยอดตำหนิที่พบ (รวม {total_defects} จุด):</b><br>
                    • {defect_str}
                </p>
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            if st.button("⏭️ ตรวจสอบชิ้นถัดไป (Inspect Next Item)", type="primary", use_container_width=True):
                st.session_state.camera_key += 1
                st.session_state.qc_stage = 'capture'
                st.session_state.should_scroll = True
                st.rerun()

    else:
        st.info("👈 **ขั้นตอน:** ถ่ายภาพชิ้นงานทางฝั่งซ้าย -> กดปุ่ม '✅ ยืนยันใช้รูปนี้ส่งตรวจ' เพื่อเริ่มต้นวิเคราะห์ผล")
