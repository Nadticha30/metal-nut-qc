import io
import sqlite3
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO

st.set_page_config(
    page_title="M.A.T.R.I.X. Nut - QC & Power BI Enterprise",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_shift(now_datetime):
    hour = now_datetime.hour
    if 8 <= hour < 16:
        return "Shift A (Day)"
    elif 16 <= hour < 24:
        return "Shift B (Evening)"
    else:
        return "Shift C (Night)"


def init_db():
    conn = sqlite3.connect("qc_metrics.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS qc_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            inspection_date TEXT,
            inspection_time TEXT,
            shift TEXT,
            status TEXT,
            total_defects INTEGER,
            crack_count INTEGER,
            scratch_count INTEGER,
            confidence_used REAL,
            processing_time_ms REAL
        )
    """)
    conn.commit()
    conn.close()


def save_log(status, total_defects, crack_cnt, scratch_cnt, conf, proc_time_ms):
    conn = sqlite3.connect("qc_metrics.db")
    cursor = conn.cursor()
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    shift_name = get_shift(now)

    cursor.execute(
        """
        INSERT INTO qc_logs (
            timestamp, inspection_date, inspection_time, shift, status, 
            total_defects, crack_count, scratch_count, confidence_used, processing_time_ms
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            ts,
            date_str,
            time_str,
            shift_name,
            status,
            total_defects,
            crack_cnt,
            scratch_cnt,
            conf,
            proc_time_ms,
        ),
    )
    conn.commit()
    conn.close()


def get_all_logs():
    conn = sqlite3.connect("qc_metrics.db")
    df = pd.read_sql_query("SELECT * FROM qc_logs ORDER BY id DESC", conn)
    conn.close()
    return df


def export_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="QC_Inspection_Logs")
    return output.getvalue()


init_db()

if "total_scanned" not in st.session_state:
    st.session_state.total_scanned = 0
if "pass_count" not in st.session_state:
    st.session_state.pass_count = 0
if "fail_count" not in st.session_state:
    st.session_state.fail_count = 0
if "qc_stage" not in st.session_state:
    st.session_state.qc_stage = "capture"
if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0
if "should_scroll" not in st.session_state:
    st.session_state.should_scroll = False

if st.session_state.should_scroll:
    st.session_state.should_scroll = False
    components.html(
        "<script>window.parent.scrollTo({top: 0, behavior:"
        " 'smooth'});</script>",
        height=0,
    )

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

    div[data-testid="stRadio"] > label p {
        color: #881337 !important;
        font-size: 18px !important;
        font-weight: 700 !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label p,
    div[data-testid="stRadio"] div[role="radiogroup"] span {
        color: #1F2937 !important;
        font-size: 17px !important;
        font-weight: 600 !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label {
        background-color: #FFFFFF !important;
        padding: 10px 16px !important;
        border-radius: 8px !important;
        border: 1.5px solid #FECDD3 !important;
        margin-right: 10px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }

    div[data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #FECDD3 !important;
        padding: 10px 14px !important;
        border-radius: 10px !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03) !important;
    }

    div[data-testid="stMetricLabel"] p {
        color: #9F1239 !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    div[data-testid="stMetricValue"] div {
        color: #881337 !important;
        font-weight: 700 !important;
        font-size: 26px !important;
    }

    .header-banner {
        background: linear-gradient(135deg, #FFFFFF 0%, #FFE4E6 100%);
        padding: 20px 25px;
        border-radius: 16px;
        border: 1px solid #FECDD3;
        box-shadow: 0 4px 15px rgba(225, 29, 72, 0.05);
        margin-bottom: 20px;
    }

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

    div.stButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 10px 16px !important;
    }

    div.stButton > button[kind="primary"] {
        background-color: #E11D48 !important;
        color: #FFFFFF !important;
        border: 1px solid #BE123C !important;
        box-shadow: 0 3px 6px rgba(225, 29, 72, 0.25) !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return YOLO("best.pt")


model = load_model()

with st.sidebar:
    st.markdown(
        "<div class='sidebar-header'>⚙️ เมนูหลักระบบ</div>", unsafe_allow_html=True
    )
    app_mode = st.radio(
        "เลือกโหมดการทำงาน:",
        [
            "🔍 ตรวจชิ้นงาน (Real-time QC)",
            "📜 ประวัติการตรวจ & Export",
            "📊 Power BI Dashboard",
        ],
    )

    if app_mode == "🔍 ตรวจชิ้นงาน (Real-time QC)":
        st.markdown("<hr style='border-color:#FBCFE8;'>", unsafe_allow_html=True)
        conf_threshold = st.slider(
            "🎯 ความไวการตรวจจับ AI (Confidence)",
            min_value=0.2,
            max_value=0.9,
            value=0.5,
            step=0.05,
        )

        st.markdown("<hr style='border-color:#FBCFE8;'>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sidebar-header'>📊 สรุปยอดการตรวจ (Shift Summary)</div>",
            unsafe_allow_html=True,
        )

        col_sb1, col_sb2 = st.columns(2)
        col_sb1.metric("จำนวนที่ตรวจ", f"{st.session_state.total_scanned} ชิ้น")

        pass_rate = (
            (st.session_state.pass_count / st.session_state.total_scanned * 100)
            if st.session_state.total_scanned > 0
            else 0
        )
        col_sb2.metric("อัตราผ่าน", f"{pass_rate:.1f}%")

        col_sb3, col_sb4 = st.columns(2)
        col_sb3.metric("🟢 PASS", f"{st.session_state.pass_count}")
        col_sb4.metric("🔴 FAIL", f"{st.session_state.fail_count}")

        if st.button("🗑️ รีเซ็ตสถิติ", use_container_width=True):
            st.session_state.total_scanned = 0
            st.session_state.pass_count = 0
            st.session_state.fail_count = 0
            st.rerun()

st.markdown("""
<div class="header-banner">
    <h1 style="color: #881337; margin:0; font-size:30px; font-weight: 700;">🔩 M.A.T.R.I.X. Nut</h1>
    <p style="color: #9F1239; margin:5px 0 0 0; font-size:15px; font-weight: 500;">Metal Automated Testing & Real-time Inspection X System</p>
</div>
""", unsafe_allow_html=True)

if app_mode == "🔍 ตรวจชิ้นงาน (Real-time QC)":
    col_cam, col_result = st.columns([1.1, 1], gap="large")

    with col_cam:
        st.markdown(
            "<h3 style='color: #881337;'>📸 1. นำเข้าภาพชิ้นงาน (Image Input)</h3>",
            unsafe_allow_html=True,
        )

        input_method = st.radio(
            "เลือกช่องทางนำเข้าภาพ:",
            ["📸 ถ่ายภาพสด (Camera)", "📁 เลือกรูปจากคลัง / อัปโหลดไฟล์"],
            horizontal=True,
        )

        img_file_buffer = None

        if input_method == "📸 ถ่ายภาพสด (Camera)":
            st.markdown("""
                <div class="focus-guide">
                    🎯 <b>คำแนะนำโฟกัส:</b> จัดวางน็อตให้อยู่กึ่งกลางกล้อง รักษาระยะห่าง 10-15 ซม. และหลีกเลี่ยงแสงสะท้อน
                </div>
                """, unsafe_allow_html=True)
            img_file_buffer = st.camera_input(
                "",
                key=f"cam_input_{st.session_state.camera_key}",
                help="กดถ่ายภาพชิ้นงานน็อตโลหะ",
            )
        else:
            st.markdown("""
                <div class="focus-guide">
                    📁 <b>คำแนะนำอัปโหลด:</b> เลือกรูปภาพจากอัลบั้มมือถือ หรือโฟลเดอร์ในคอมพิวเตอร์ (.jpg, .jpeg, .png)
                </div>
                """, unsafe_allow_html=True)
            img_file_buffer = st.file_uploader(
                "เลือกรูปภาพชิ้นงานน็อต",
                type=["jpg", "jpeg", "png"],
                key=f"file_uploader_{st.session_state.camera_key}",
            )

        if img_file_buffer is not None:
            st.image(
                img_file_buffer,
                caption="📷 ตัวอย่างภาพถ่ายเตรียมส่งตรวจ",
                use_container_width=True,
            )

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button(
                    "✅ ยืนยันใช้รูปนี้ส่งตรวจ",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.qc_stage = "analyze"
                    st.rerun()
            with col_btn2:
                if st.button("🔄 เลือก/ถ่ายรูปใหม่", use_container_width=True):
                    st.session_state.camera_key += 1
                    st.session_state.qc_stage = "capture"
                    st.session_state.should_scroll = True
                    st.rerun()

    with col_result:
        st.markdown(
            "<h3 style='color: #881337;'>📊 2. ผลการวิเคราะห์และตรวจสอบ (QC Analysis)</h3>",
            unsafe_allow_html=True,
        )

        if img_file_buffer is not None and st.session_state.qc_stage == "analyze":
            with st.spinner("🔍 AI กำลังประมวลผลวิเคราะห์จุดบกพร่อง..."):
                start_time = time.time()
                bytes_data = img_file_buffer.getvalue()
                cv2_img = cv2.imdecode(
                    np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR
                )

                results = model.predict(
                    source=cv2_img, conf=conf_threshold, verbose=False
                )
                res = results[0]
                proc_time = round((time.time() - start_time) * 1000, 2)
                total_defects = len(res.boxes)

                class_ids = res.boxes.cls.cpu().numpy().astype(int)
                class_names = [model.names[i] for i in class_ids]
                counts = Counter(class_names)

                crack_cnt = counts.get("crack", 0)
                scratch_cnt = counts.get("scratch", 0)
                status = "PASS" if total_defects == 0 else "FAIL"

                save_log(
                    status,
                    total_defects,
                    crack_cnt,
                    scratch_cnt,
                    conf_threshold,
                    proc_time,
                )

                st.session_state.total_scanned += 1

                if status == "PASS":
                    st.session_state.pass_count += 1
                    annotated_frame = cv2_img.copy()
                    cv2.putText(
                        annotated_frame,
                        "QC: PASS (GOOD)",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3,
                    )

                    st.markdown(
                        '<div class="status-pass">🟢 สถานะชิ้นงาน: PASS (ผ่านเกณฑ์)</div>',
                        unsafe_allow_html=True,
                    )
                    st.success("✨ ชิ้นงานสมบูรณ์แบบ ไม่พบรอยแตกร้าวหรือรอยขีดข่วน")
                else:
                    st.session_state.fail_count += 1
                    annotated_frame = res.plot()
                    cv2.putText(
                        annotated_frame,
                        f"QC: FAIL ({total_defects})",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3,
                    )

                    st.markdown(
                        f'<div class="status-fail">🔴 สถานะชิ้นงาน: FAIL (พบตำหนิ {total_defects} จุด)</div>',
                        unsafe_allow_html=True,
                    )

                    m1, m2 = st.columns(2)
                    m1.metric("💥 รอยแตกร้าว (Crack)", f"{crack_cnt} จุด")
                    m2.metric("⚡ รอยขีดข่วน (Scratch)", f"{scratch_cnt} จุด")

                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(
                    frame_rgb, caption="ภาพผลการวิเคราะห์จาก AI", use_container_width=True
                )

                st.markdown(f"""
                    <div style='background-color: #FFFFFF; border: 1px solid #FECDD3; border-radius: 12px; padding: 16px; margin-top: 12px; margin-bottom: 15px;'>
                        <h4 style='color: #881337; margin: 0 0 8px 0;'>📋 สรุปรายละเอียดการตรวจ</h4>
                        <p style='margin: 0; font-size: 14px;'>
                            <b>เวลาประมวลผล:</b> {proc_time} ms | <b>บันทึกเข้า Database:</b> สำเร็จเรียบร้อย
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<hr>", unsafe_allow_html=True)
                if st.button(
                    "⏭️ ตรวจสอบชิ้นถัดไป (Inspect Next Item)",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.camera_key += 1
                    st.session_state.qc_stage = "capture"
                    st.session_state.should_scroll = True
                    st.rerun()
        else:
            st.info(
                "👈 **ขั้นตอน:** เลือกรูปภาพทางฝั่งซ้าย -> กดปุ่ม '✅ ยืนยันใช้รูปนี้ส่งตรวจ' เพื่อเริ่มต้นวิเคราะห์ผล"
            )

elif app_mode == "📜 ประวัติการตรวจ & Export":
    st.title("📜 ประวัติการตรวจสอบชิ้นงานย้อนหลัง")
    st.caption(
        "ระบบบันทึกประวัติรายละเอียด แยกตามวันที่ เวลา กะการทำงาน (Shift) และประเภทความเสียหาย พร้อมส่งออกไฟล์"
    )

    df_logs = get_all_logs()

    if not df_logs.empty:
        st.markdown("### 📥 ดาวน์โหลดรายงานประวัติการทำงาน")
        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            excel_bytes = export_to_excel(df_logs)
            st.download_button(
                label="📗 ดาวน์โหลดรายงานไฟล์ Excel (.xlsx)",
                data=excel_bytes,
                file_name=f"QC_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col_ex2:
            csv_bytes = df_logs.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                label="📄 ดาวน์โหลดรายงานไฟล์ CSV (.csv)",
                data=csv_bytes,
                file_name=f"QC_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("### 📋 ตารางบันทึกข้อมูลย้อนหลังทั้งหมด")
        st.dataframe(df_logs, use_container_width=True, height=450)
    else:
        st.info("ยังไม่มีข้อมูลประวัติการตรวจในระบบ")

elif app_mode == "📊 Power BI Dashboard":
    st.title("📊 Executive Dashboard (Power BI Integrated)")
    st.caption("แดชบอร์ดสรุปผลเชิงบริหาร เชื่อมโยงข้อมูลประวัติการตรวจแบบ Interactive")

    POWER_BI_EMBED_URL = "https://app.powerbi.com/view?r=eyJrIjoiZjUxZjQ4NDItOWQ3NS00NDIzLTg2ZDctOGI1OGI3NGI1ZWIzIiwidCI6IjhhOWQzNmYwLTVjOWEtNGU0MC1hYzVkLTQxZmY4M2ZjZTA2NCIsImMiOjEwfQ%3D%3D"

    # แสดงผล Power BI iframe แบบกว้างเต็มความจุหน้าจอ (100% responsive width)
    st.markdown(
        f"""
        <iframe 
            src="{POWER_BI_EMBED_URL}" 
            width="100%" 
            height="750" 
            style="border:none; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);"
            allowFullScreen="true">
        </iframe>
        """,
        unsafe_allow_html=True,
    )
