from flask import Flask, render_template, request
from utils.ocr import extract_text_from_image
from utils.pdf_reader import extract_text_from_pdf
from utils.text_cleaner import clean_text
from models.summarizer import summarize_report

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    extracted_text = ""
    summary = ""
    if request.method == "POST":
        uploaded_file = request.files.get("file")

        if uploaded_file:
            # Extract text from PDF or Image
            if uploaded_file.filename.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = extract_text_from_image(uploaded_file)

            # Clean the text
            extracted_text = clean_text(extracted_text)

            # Generate point-wise summary
            summary = summarize_report(extracted_text)

    return render_template(
        "index.html",
        extracted_text=extracted_text,
        summary=summary
    )

if __name__ == "__main__":
    app.run(debug=True)
