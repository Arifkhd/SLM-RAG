import streamlit as st

from scripts.ocr import extract_text_from_image
from scripts.pdf_reader import extract_text_from_pdf
from scripts.text_cleaner import clean_text
from scripts.json_loader import load_json
from models.summarizer import summarize_report
from scripts.recommender import recommend_specialist


# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Radiology AI Assistant",
    layout="wide"
)

st.title("🧠 Radiology Report Summarizer & Specialist Recommender")
st.write("Upload a radiology report (PDF / Image) to get a summary and specialist recommendation.")

# ------------------ LOAD SPECIALIST DATA ------------------
try:
    specialists = load_json("specialist_doctor/radiology_healthcare_dataset.json")
except Exception as e:
    st.error("❌ Failed to load specialist dataset")
    st.exception(e)
    st.stop()

# Optional: Debug view
with st.expander("🔍 View Loaded Specialist Data"):
    st.write(specialists)

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader(
    "📤 Upload Radiology Report",
    type=["pdf", "png", "jpg", "jpeg"]
)

if uploaded_file:
    st.info("⏳ Processing report...")

    # ------------------ TEXT EXTRACTION ------------------
    try:
        if uploaded_file.type == "application/pdf":
            raw_text = extract_text_from_pdf(uploaded_file)
        else:
            raw_text = extract_text_from_image(uploaded_file)
    except Exception as e:
        st.error("❌ Failed to extract text from file")
        st.exception(e)
        st.stop()

    if not raw_text.strip():
        st.warning("⚠️ No readable text found in the uploaded file.")
        st.stop()

    # ------------------ CLEAN TEXT ------------------
    cleaned_text = clean_text(raw_text)

    st.subheader("📄 Extracted Text")
    st.text_area("Extracted Content", cleaned_text, height=200)

    # ------------------ SUMMARIZATION ------------------
    try:
        summary = summarize_report(cleaned_text)
    except Exception as e:
        st.error("❌ Error during summarization")
        st.exception(e)
        st.stop()

    st.subheader("📝 Report Summary")
    st.success(summary)

    # ------------------ SPECIALIST RECOMMENDATION ------------------
    try:
        specialist, confidence = recommend_specialist(summary, specialists)
    except Exception as e:
        st.error("❌ Error during specialist recommendation")
        st.exception(e)
        st.stop()

    st.subheader("👨‍⚕️ Recommended Specialist")
    st.info(f"**{specialist}**  \nConfidence Score: **{confidence}**")

# ------------------ DISCLAIMER ------------------
st.caption("⚠️ This AI system is for clinical decision support only. Not a medical diagnosis.")
