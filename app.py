import streamlit as st
import pandas as pd
import os
import requests
import json
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
BMI_API_URL = "https://hotelmerdekayapen.my.id/bmi/api/bmi"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
LLM_URL = "https://api.groq.com/openai/v1/chat/completions"

# Page config
st.set_page_config(
    page_title="𝐇𝐞𝐚𝐥𝐭𝐡𝐓𝐫𝐚𝐜𝐤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling (dari app.py)
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #ffffff;
        font-size: 16px;
        padding: 12px 20px;
        margin: 5px 0;
        cursor: pointer;
        border-radius: 8px;
        transition: all 0.3s;
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p:hover {
        background-color: #2d2d2d;
    }
    
    .main-title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #666;
        margin-bottom: 30px;
    }
    
    .info-box {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1e90ff;
        margin: 20px 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 20px 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 20px 0;
    }
    
    .stButton>button {
        width: 100%;
        background-color: #1e90ff;
        color: white;
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
        border: none;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #1873cc;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "bmi_result" not in st.session_state:
    st.session_state.bmi_result = None


def generate_llm_response(user_msg, bmi_data=None, chat_history=[]):
    """Generate LLM response dengan konteks BMI data"""
    
    # Buat konteks dari data BMI jika ada
    context_text = ""
    if bmi_data:
        context_text = f"""
Data Kesehatan Pasien:
- BMI: {bmi_data.get('bmi', 'N/A')}
- Kategori: {bmi_data.get('kategori', 'N/A')}
- Berat Ideal: {bmi_data.get('berat_ideal', 'N/A')} kg
- Jarak ke Ideal: {bmi_data.get('jarak_ke_ideal', 'N/A')}
- Prediksi Model: {bmi_data.get('prediksi_model', 'N/A')}
- Sumber: {bmi_data.get('source', 'N/A')}
"""
    
    rules_prompt = (
        "🌟 Hai! Kamu adalah **HealthBot**, Asisten AI kesehatan yang ramah, murah senyum 😄, sopan 🙏, dan komunikatif ✨.\n\n"
        "🎯 Tugas utama kamu adalah membantu memberikan saran kesehatan, khususnya terkait **BMI (Body Mass Index)**, "
        "berat badan ideal, nutrisi, olahraga, dan gaya hidup sehat. Semua dijelaskan dengan gaya yang santai 🧋, "
        "tapi tetap profesional dan informatif 🧠.\n\n"
        "🗣️ Gaya ngobrol kamu: ringan, hangat, bisa bercanda 🤪, support curhat atau galau 🥲, "
        "tapi selalu fokus membantu user jadi lebih sehat dan memahami kondisi tubuh mereka.\n\n"
        "✨ Kalau diminta saran kesehatan, berikan saran yang praktis dan mudah diikuti. "
        "Jika ada data BMI, berikan analisis dan rekomendasi yang spesifik!\n\n"
        "📌 **PENTING**:\n"
        "• Berikan saran yang aman dan umum (bukan diagnosis medis).\n"
        "• Selalu sarankan konsultasi dengan dokter untuk kondisi serius.\n"
        "• Bersikap supportif dan motivatif untuk perubahan positif.\n"
        "• Fokus pada solusi praktis: diet, olahraga, gaya hidup.\n\n"
        "🧠 Tetap semangat bantu user dengan **senyum virtual 😁** dan vibes positif! "
        "Jadilah partner AI yang humble, helpful, dan data-driven 💡.\n"
    )
    
    messages = [
        {"role": "system", "content": rules_prompt},
        *[
            {"role": item["role"], "content": item["content"]}
            for item in chat_history[-8:] if item["role"] in ["user", "assistant"]
        ],
    ]
    
    if context_text:
        messages.append({"role": "system", "content": f"Informasi:\n{context_text}"})
    
    messages.append({"role": "user", "content": user_msg})
    
    try:
        if not LLM_URL or not MODEL or not GROQ_API_KEY:
            missing = [k for k, v in {
                "URL_EMBEDDING_MODEL": LLM_URL, 
                "MODEL_NAME": MODEL, 
                "GROQ_API_KEY": GROQ_API_KEY
            }.items() if not v]
            raise RuntimeError(f"Missing required env: {', '.join(missing)}")
        
        response = requests.post(
            LLM_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2024,
                "top_p": 0.9,
            },
            timeout=32,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Maaf, AI sedang bermasalah: {e}"


def predict_bmi(tinggi, berat, usia, jenis_kelamin):
    """Panggil API BMI untuk mendapatkan prediksi"""
    try:
        payload = {
            "tinggi": tinggi,
            "berat": berat,
            "usia": usia,
            "jenis_kelamin": jenis_kelamin
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(BMI_API_URL, json=payload, headers=headers, timeout=10)

        if response.status_code == 200:
            return response.json(), None
        else:
            return None, f"API Error: {response.status_code} - {response.text}"

    except requests.exceptions.Timeout:
        return None, "Koneksi timeout. Silakan coba lagi."
    except requests.exceptions.ConnectionError:
        return None, "Tidak dapat terhubung ke server. Periksa koneksi internet Anda."
    except Exception as e:
        return None, f"Terjadi kesalahan: {str(e)}"


# Sidebar Navigation
with st.sidebar:
    st.markdown("### HealthTrack")
    st.markdown("---")

    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

    if st.button("⏲ Prediksi", use_container_width=True):
        st.session_state.page = "prediksi"
        st.rerun()

    if st.button("🛈 About", use_container_width=True):
        st.session_state.page = "about"
        st.rerun()


# ==================== HOME PAGE ====================
if st.session_state.page == "home":

    st.markdown("<h1 class='main-title'>HealthTrack</h1>", unsafe_allow_html=True)

    # Banner
    banner_path = "assets/images/healthtrack_banner.png"
    if os.path.exists(banner_path):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(banner_path, use_container_width=True)
    else:
        st.info("🖼 Tambahkan gambar banner: assets/images/healthtrack_banner.png")

    st.markdown("---")
    st.markdown("## Selamat datang di HealthTrack")
    st.markdown("""
    HealthTrack membantu kamu mengidentifikasi kondisi berat badan, memprediksi perubahan berat badan, dan mendapatkan rekomendasi nutrisi & aktivitas secara personal.

    HealthTrack dirancang untuk mendukung gaya hidup sehat dengan pendekatan digital dan interaktif, agar kamu bisa memantau kondisi tubuh kapan saja dan di mana saja.
    """)
    st.markdown("---")

    # Standar BMI
    st.markdown("## TAHUKAH KAMU BERAPA STANDAR BMI")

    st.info("""
### Standar Kategori BMI (Body Mass Index)

BMI adalah indikator untuk mengetahui apakah berat badan kamu:
- **Kurus (Underweight)**
- **Normal (Ideal)**
- **Overweight**
- **Obesitas**

Rumus perhitungan BMI:  
**BMI = Berat Badan (kg) / (Tinggi Badan (m))²**
""")

    # Gambar kategori BMI
    chart_path = "assets/images/kategori_BMI.jpg"

    if os.path.exists(chart_path):
        st.image(chart_path, caption="Kategori Standar BMI", use_container_width=True)
    else:
        st.info("📊 Tambahkan gambar: assets/images/kategori_BMI.jpg")

    # Tabel standar BMI
    st.markdown("### Standar BMI")
    st.markdown("""
| BMI | Kategori |
|------|----------|
| < 18.5 | **Underweight (Kurus)** |
| 18.5 – 24.9 | **Normal (Ideal)** |
| 25 – 29.9 | **Overweight (Kelebihan Berat Badan)** |
| ≥ 30 | **Obesitas** |
""")

    st.markdown("---")
    _, colBtn, _ = st.columns([1, 2, 1])
    with colBtn:
        if st.button("🔍 Mulai Prediksi Sekarang", key="cta_button"):
            st.session_state.page = "prediksi"
            st.rerun()


# ==================== PREDIKSI PAGE ====================
elif st.session_state.page == "prediksi":

    st.markdown("<h1 class='main-title'>📊 Prediksi Kondisi Berat Badan</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Masukkan data diri kamu untuk mengetahui kondisi berat badan</p>", unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        age = st.number_input("Usia (tahun)", min_value=10, max_value=100, value=25)

    with col2:
        height = st.number_input("Tinggi Badan (cm)", min_value=100, max_value=220, value=165)
        weight = st.number_input("Berat Badan (kg)", min_value=30, max_value=200, value=60)

    st.markdown("---")

    if st.button("Prediksi Sekarang", key="predict_button"):

        gender_api = "Male" if gender == "Laki-laki" else "Female"

        with st.spinner("Memproses prediksi..."):
            api_response, error = predict_bmi(height, weight, age, gender_api)

        if error:
            st.error(f"❌ {error}")

        elif api_response:
            # Simpan hasil ke session state
            st.session_state.bmi_result = api_response
            
            result = api_response.get("kategori", api_response.get("prediction", api_response.get("result", "Unknown")))
            bmi_value = api_response.get("bmi", 0)
            berat_ideal = api_response.get("berat_ideal", "N/A")
            jarak = api_response.get("jarak_ke_ideal", "N/A")

            st.markdown("---")
            st.markdown("## Hasil Prediksi")

            # Status box berdasarkan kategori (UI dari app.py)
            if result.lower() in ["underweight", "kurus"]:
                st.markdown("<div class='info-box'>📉 KURUS (Underweight)</div>", unsafe_allow_html=True)
            elif result.lower() in ["normal", "ideal"]:
                st.markdown("<div class='success-box'>✅ IDEAL (Normal Weight)</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='warning-box'>🔴 OBESITAS (Overweight/Obesity)</div>", unsafe_allow_html=True)

            st.markdown("---")

            # Gauge Chart dengan Plotly
            if bmi_value < 18.5:
                bar_color = "#87CEEB"
                category_text = "Underweight"
            elif 18.5 <= bmi_value < 25:
                bar_color = "#90EE90"
                category_text = "Normal"
            elif 25 <= bmi_value < 30:
                bar_color = "#FFD700"
                category_text = "Overweight"
            elif 30 <= bmi_value < 35:
                bar_color = "#FFA07A"
                category_text = "Obesity"
            else:
                bar_color = "#FF6B6B"
                category_text = "Extremely obese"

            # Membuat gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bmi_value,
                domain={'x': [0, 1], 'y': [0, 1]},
                number={'suffix': "", 'font': {'size': 50, 'color': '#1a1a1a'}},
                title={'text': "Your BMI is", 'font': {'size': 20, 'color': '#666'}},
                gauge={
                    'axis': {'range': [None, 40], 'tickwidth': 1, 'tickcolor': "darkgray"},
                    'bar': {'color': bar_color, 'thickness': 0.3},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 18.5], 'color': '#E6F4FA'},
                        {'range': [18.5, 25], 'color': '#E8F5E9'},
                        {'range': [25, 30], 'color': '#FFF9E6'},
                        {'range': [30, 35], 'color': '#FFE8D6'},
                        {'range': [35, 40], 'color': '#FFE0E0'}
                    ],
                    'threshold': {
                        'line': {'color': bar_color, 'width': 4},
                        'thickness': 0.75,
                        'value': bmi_value
                    }
                }
            ))

            fig.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=80, b=20),
                paper_bgcolor="white",
                font={'family': "Arial"}
            )

            # Tampilkan dalam 2 kolom
            col_chart, col_info = st.columns([1, 1])

            with col_chart:
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(f"""
                <div style='text-align: center; color: #999; font-size: 14px; margin-top: -20px;'>
                    {height}cm | {weight}kg | {gender}
                </div>
                """, unsafe_allow_html=True)

            with col_info:
                st.markdown("### 📋 BMI Fitness Chart")
                st.markdown("---")

                categories = [
                    ("🔵 Underweight", "<18.5", bmi_value < 18.5),
                    ("🟢 Normal", "18.5-24.9", 18.5 <= bmi_value < 25),
                    ("🟡 Overweight", "25-29.9", 25 <= bmi_value < 30),
                    ("🟠 Obesity", "30-34.9", 30 <= bmi_value < 35),
                    ("🔴 Extremely obese", ">35", bmi_value >= 35)
                ]

                for label, range_val, is_current in categories:
                    if is_current:
                        st.markdown(f"""
                        <div style='background-color: {bar_color}; padding: 12px; border-radius: 8px; margin: 8px 0; display: flex; justify-content: space-between; font-weight: bold;'>
                            <span>{label}</span>
                            <span>{range_val}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='padding: 10px; border-radius: 8px; margin: 8px 0; display: flex; justify-content: space-between; color: #999;'>
                            <span>{label}</span>
                            <span style='color: #bbb;'>{range_val}</span>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📊 Informasi Detail")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("BMI Kamu", f"{bmi_value:.1f}")
            with col2:
                st.metric("Tinggi Badan", f"{height} cm")
            with col3:
                st.metric("Berat Badan", f"{weight} kg")
            with col4:
                st.metric("Berat Ideal", f"{berat_ideal} kg")

            st.markdown("---")

            # ========== TAMBAHAN: SOLUSI AI ==========
            st.markdown("## 💡 Solusi & Rekomendasi Kesehatan dari AI")

            with st.spinner("🤖 AI sedang menganalisis kondisi Anda dan memberikan solusi terbaik..."):
                ai_question = f"""Berdasarkan hasil analisis BMI berikut:
- Berat Badan: {weight} kg
- Tinggi Badan: {height} cm
- Usia: {age} tahun
- Jenis Kelamin: {gender}
- BMI: {bmi_value}
- Kategori: {result}
- Berat Ideal: {berat_ideal} kg
- Jarak ke berat ideal: {jarak}

Tolong berikan solusi dan rekomendasi kesehatan yang lengkap, praktis, dan personal untuk kondisi saya, meliputi:

1. 📊 **Analisis Kondisi Kesehatan**: Jelaskan kondisi BMI saya saat ini dan dampaknya
2. 🎯 **Target Realistis**: Berapa berat badan yang harus dicapai dan dalam waktu berapa lama
3. 🍎 **Rekomendasi Pola Makan**: Menu makanan sehat yang cocok untuk kondisi saya
4. 🏃 **Program Olahraga**: Jenis dan durasi olahraga yang tepat untuk saya
5. 💪 **Tips Gaya Hidup**: Kebiasaan sehat yang mudah diterapkan sehari-hari
6. ⚠️ **Hal Penting**: Yang perlu diperhatikan dan dihindari

Jelaskan dengan detail, ramah, dan mudah dipahami ya! 😊"""

                ai_solution = generate_llm_response(
                    ai_question,
                    bmi_data=api_response,
                    chat_history=[]
                )

            # Tampilkan solusi dari AI
            st.markdown(ai_solution)


# ==================== ABOUT PAGE ====================
elif st.session_state.page == "about":
    st.markdown("<h1 class='main-title'>Tentang HealthTrack</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Solusi Digital untuk Kesehatan Anda</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.info("""
    Proyek ini menghadirkan solusi digital interaktif dalam bidang kesehatan, 
    untuk membantu pengguna memahami status berat badan berdasarkan indeks BMI.
    """)

    st.markdown("## Teknologi Yang Digunakan")
    st.markdown("""
    - Streamlit
    - Python
    - API
    - LLM
    """)

    st.markdown("---")

    st.success("Link Github: ")
