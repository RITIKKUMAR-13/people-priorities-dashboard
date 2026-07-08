import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from deep_translator import GoogleTranslator
import speech_recognition as sr
from PIL import Image
import cv2
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(layout="wide")
 

# ------------------- Custom Background + Floating Button CSS -------------------


page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #0f0f0f, #2c2f33);
    color: white;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
textarea, input {
    color: white !important;
    background-color: #1e1e1e !important;
    font-size: 16px !important;
}
[data-testid="stFileUploader"] div {
    color: white !important;
}
.summary-card {
    background-color: #1e1e1e;
    padding: 20px;
    border-radius: 10px;
    color: white;
    font-size: 16px;
    font-family: Arial, sans-serif;
}
/* Floating chat button */
button[data-baseweb="button"][id="chat_icon"] {
    position: fixed;
    bottom: 30px;
    right: 30px;
    background-color: #1e1e1e;
    color: white;
    border-radius: 50%;
    padding: 25px;
    font-size: 28px;
    z-index: 9999;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.5);
    transition: all 0.3s ease-in-out;
    animation: pulse 2s infinite;
}
/* Glow effect on hover */
button[data-baseweb="button"][id="chat_icon"]:hover {
    background-color: #2c2f33;
    box-shadow: 0px 0px 20px rgba(255,255,255,0.8);
    transform: scale(1.1);
}
/* Pulse animation */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,255,255,0.7); }
    70% { box-shadow: 0 0 0 20px rgba(255,255,255,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,255,255,0); }
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    table {
        width: 100% !important;
        word-wrap: break-word;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------- Title -------------------
st.title("📊 People's Priorities - AI Dashboard")
st.subheader("🌍 Multilingual, Voice & Photo Feedback Analyzer")



# ------------------- Sentiment Analysis -------------------
st.header("📝 Sentiment Analysis")
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
user_input = st.text_area("Enter citizen feedback:", key="sentiment_input")
if st.button("Analyze Sentiment", key="sentiment_btn"):
    if user_input.strip():
        result = sentiment_model(user_input)[0]
        st.success(f"Feedback: {user_input}")
        st.info(f"Sentiment: {result['label']} (Confidence: {round(result['score'],2)})")
    else:
        st.warning("Please enter some feedback!")
        st.markdown("<div class='ai-glow'>🤖 AI is analyzing your request...</div>", unsafe_allow_html=True)


# ------------------- Multilingual Translation -------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.header("🌐 Multilingual Feedback Translator")

user_input = st.text_area("Enter feedback in Hindi/other language:")

if user_input:
    try:
        translated_text = GoogleTranslator(source='auto', target='en').translate(user_input)
        st.success(f"Translated Feedback (English): {translated_text}")
    except Exception as e:
        st.error(f"Translation failed: {e}")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------- Recurring Themes Analyzer -------------------
# ✅ Example Recurring Themes Analyzer
data = {
    "Feedback": [
        "Need school upgrade",
        "Hospital required",
        "Road repair needed",
        "Vocational training center",
        "More teachers needed",
        "Better healthcare facilities",
        "No issues mentioned"
    ],
    "Count": [2, 0, 0, 0, 0, 0, 1]
}

df = pd.DataFrame(data)

st.subheader("Recurring Themes Analyzer")
# ✅ Responsive table for both mobile & laptop
st.dataframe(df, use_container_width=True)

# ------------------- Voice Input -------------------
st.header("🎤 Voice Feedback Input")
if st.button("Record Voice Feedback", key="voice_btn"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Speak now...")
        audio = r.listen(source)
    try:
        text = r.recognize_google(audio, language="hi-IN")
        st.success(f"You said (Hindi): {text}")
    except:
        try:
            text = r.recognize_google(audio, language="en-IN")
            st.success(f"You said (English): {text}")
        except Exception as e:
            st.error("Could not recognize voice: " + str(e))

# ------------------- Photo Input -------------------
st.header("📷 Photo Feedback Input")
uploaded_file = st.file_uploader("Upload a photo (jpg/png)", type=["jpg", "png"], key="photo_uploader")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Photo", use_column_width=True)
    img_cv = np.array(image)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    st.image(gray, caption="Processed (Gray Scale)", use_column_width=True)

# ------------------- Executive Summary -------------------
summary_text = "Over the last 48 hours, there has been a 15% spike in complaints regarding school infrastructure in Dadri. Meanwhile, infrastructure issues in Ghaziabad remain steady, primarily focused on non-functioning streetlights."
st.markdown(f"<div class='summary-card'><b>🤖 AI Executive Summary</b><br>{summary_text}</div>", unsafe_allow_html=True)

# ------------------- Dashboard -------------------
st.header("📈 MP Dashboard")
data = {
    "Feedback": [
        "Need school upgrade",
        "Hospital required",
        "Road repair needed",
        "Vocational training centre",
        "More teachers in schools",
        "Better healthcare facilities",
        "Fix broken streetlights"
    ],
    "Category": ["Education", "Health", "Infrastructure", "Education", "Education", "Health", "Infrastructure"],
    "Location": ["Dadri", "Dadri", "Noida", "Noida", "Dadri", "Ghaziabad", "Ghaziabad"]
}
df = pd.DataFrame(data)
st.subheader("Citizen Feedback Data")
df = pd.DataFrame(data)
# ✅ Responsive table for mobile + laptop
st.dataframe(df, use_container_width=True)

fig_bar = px.bar(df, x="Category", title="Feedback by Category", color="Category")
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_bar, key="bar_chart", use_container_width=True)
with col2:
    fig_pie = px.pie(df, names="Category", title="Feedback Distribution")
    st.plotly_chart(fig_pie, key="pie_chart", use_container_width=True)


# ------------------- Recommendation Engine -------------------
st.header("⭐ Recommendation Engine")
urgency = [3,5,4,2,4,5,3]
df["Urgency"] = urgency
category_counts = df["Category"].value_counts().to_dict()
df["Frequency"] = df["Category"].map(category_counts)
df["PriorityScore"] = df["Frequency"] + df["Urgency"]
df_sorted = df.sort_values(by="PriorityScore", ascending=False)
st.subheader("📊 Ranked Priority Projects")
# ✅ Responsive table for mobile + laptop
st.dataframe(df_sorted[["Feedback","Category","Urgency","Frequency","PriorityScore"]],
             use_container_width=True)

# ------------------- Copilot Chatbot (Floating Button) -------------------
# ✅ Embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("text-generation", model="facebook/bart-large-cnn")
summary = summarizer("Summarize: Your long text here", max_length=130, min_length=30, do_sample=False)
print(summary[0]['generated_text'])



# ------------------- Streamlit Chatbot -------------------

# Toggle state
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False

# Floating button
if st.button("💬", key="chat_icon"):
    st.session_state.show_chat = not st.session_state.show_chat

# Chatbot window
if st.session_state.show_chat:
    st.markdown("### 💬 Copilot Chatbot")
    query = st.text_input("Ask your data:", key="chat_query")
    if query:
        feedback_embeddings = embedder.encode(df["Feedback"].tolist())
        query_embedding = embedder.encode([query])
        sims = cosine_similarity(query_embedding, feedback_embeddings).flatten()
        top_idx = sims.argsort()[-3:][::-1]
        context = df.iloc[top_idx]["Feedback"].tolist()
        prompt = f"Citizen feedback: {context}. Question: {query}"

        # ✅ Use generated_text instead of summary_text
        summary = summarizer(prompt, max_length=120, min_length=40, do_sample=False)[0]["generated_text"]

        st.markdown(
            f"<div class='summary-card'><h4>🤖 Copilot Answer</h4>{summary}</div>",
            unsafe_allow_html=True
        )


# ------------------- Predictive AI (Anomalies & Future Trends) -------------------
st.header("🔮 Predictive AI - Hotspot Indicator")

# Example complaint data with timestamps
complaints_data = pd.DataFrame({
    "Feedback": [
        "Broken water pipeline",
        "Broken water pipeline",
        "Broken water pipeline",
        "Streetlight not working",
        "Hospital required",
        "Hospital required"
    ],
    "Location": ["Sector 12","Sector 12","Sector 12","Sector 15","Dadri","Dadri"],
    "Timestamp": pd.to_datetime([
        "2026-07-07 10:00:00",
        "2026-07-07 11:00:00",
        "2026-07-07 12:00:00",
        "2026-07-07 09:00:00",
        "2026-07-07 10:30:00",
        "2026-07-07 11:00:00"
    ])
})

# Rule: If same location + same issue >= 3 times in short window → Escalation Risk
risk_flags = []
for idx, row in complaints_data.iterrows():
    subset = complaints_data[
        (complaints_data["Location"] == row["Location"]) &
        (complaints_data["Feedback"] == row["Feedback"])
    ]
    if len(subset) >= 3:
        risk_flags.append("⚠️ Escalation Risk")
    else:
        risk_flags.append("Normal")
complaints_data["RiskLevel"] = risk_flags

st.subheader("Predictive Hotspot Indicator")
# ✅ Responsive table for mobile + laptop
st.dataframe(complaints_data[["Feedback","Location","Timestamp","RiskLevel"]],
             use_container_width=True)

# Visualization
fig_hotspot = px.scatter(
    complaints_data,
    x="Timestamp",
    y="Location",
    color="RiskLevel",
    size=[15 if r=="⚠️ Escalation Risk" else 8 for r in complaints_data["RiskLevel"]],
    title="Predictive Hotspot Indicator (Escalation Risks)"
)
st.plotly_chart(fig_hotspot, use_container_width=True)

