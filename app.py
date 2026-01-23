import os
import re
from flask import Flask, render_template, request, redirect, url_for, session, flash
from dotenv import load_dotenv
from supabase import create_client

from utils.ocr import extract_text_from_image
from utils.pdf_reader import extract_text_from_pdf
from utils.text_cleaner import clean_text
from models.summarizer import summarize_report

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-me")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY


# -----------------------------
# HELPERS
# -----------------------------
def is_logged_in() -> bool:
    return "user" in session


# -----------------------------
# HOME (OPEN APP -> SIGN IN FIRST)
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    # If user NOT logged in → redirect to Sign In page
    if not is_logged_in():
        return redirect(url_for("signin"))

    # If logged in → show dashboard (index.html)
    return render_template("index.html", user=session.get("user"))


# -----------------------------
# AUTH: SIGN UP (Name + Email + Password)
# -----------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    # If already logged in, go to dashboard
    if is_logged_in():
        return redirect(url_for("home"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not name or not email or not password:
            flash("All fields are required.", "error")
            return render_template("signup.html", user=session.get("user"))

        try:
            supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {"data": {"name": name}}
            })

            flash("Account created! Please sign in (verify email if required).", "success")
            return redirect(url_for("signin"))

        except Exception as e:
            flash(f"Signup failed: {str(e)}", "error")
            return render_template("signup.html", user=session.get("user"))

    return render_template("signup.html", user=session.get("user"))


# -----------------------------
# AUTH: SIGN IN (Email + Password)
# -----------------------------
@app.route("/signin", methods=["GET", "POST"])
def signin():
    # If already logged in, go to dashboard
    if is_logged_in():
        return redirect(url_for("home"))

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not email or not password:
            flash("Email and password are required.", "error")
            return render_template("signin.html", user=session.get("user"))

        try:
            res = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
            })

            user = res.user
            sess = res.session

            session["access_token"] = sess.access_token
            session["refresh_token"] = sess.refresh_token
            session["user"] = {
                "id": user.id,
                "email": user.email,
                "name": (user.user_metadata or {}).get("name", "User"),
            }

            flash("Signed in successfully.", "success")
            return redirect(url_for("home"))

        except Exception:
            flash("Invalid email or password (or email not verified).", "error")
            return render_template("signin.html", user=session.get("user"))

    return render_template("signin.html", user=session.get("user"))


# -----------------------------
# AUTH: LOGOUT
# -----------------------------
@app.route("/logout", methods=["GET"])
def logout():
    session.clear()
    flash("Logged out successfully.", "success")
    return redirect(url_for("signin"))


# -----------------------------
# REPORT SUMMARIZER (PROTECTED)
# -----------------------------
@app.route("/report-summarizer", methods=["GET", "POST"])
def report_summarizer():
    if not is_logged_in():
        flash("Please sign in to use Report Summarizer.", "error")
        return redirect(url_for("signin"))

    extracted_text = ""
    summary = []

    # Reset results
    if request.method == "POST" and "reset" in request.form:
        return render_template(
            "report_summarizer.html",
            extracted_text="",
            summary=[],
            user=session.get("user")
        )

    if request.method == "POST":
        uploaded_file = request.files.get("report_file")

        if uploaded_file and uploaded_file.filename:
            filename = uploaded_file.filename.lower()

            # Extract text from PDF or image
            if filename.endswith(".pdf"):
                extracted_text = extract_text_from_pdf(uploaded_file)
            else:
                extracted_text = extract_text_from_image(uploaded_file)

            extracted_text = clean_text(extracted_text)

            # Summarize
            summary = summarize_report(extracted_text)

            # Ensure list of points (fix char-by-char issue)
            if summary is None:
                summary = []
            elif isinstance(summary, str):
                s = summary.strip()
                lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
                if len(lines) <= 1:
                    parts = re.split(r"(?:\r?\n)+|\s*(?:\d+[\)\.\-]|[-•])\s+", s)
                    summary = [p.strip() for p in parts if p.strip()]
                else:
                    summary = [re.sub(r"^\s*(?:\d+[\)\.\-]|[-•])\s*", "", ln).strip() for ln in lines]
            else:
                summary = [str(x).strip() for x in summary if str(x).strip()]

    return render_template(
        "report_summarizer.html",
        extracted_text=extracted_text,
        summary=summary,
        user=session.get("user")
    )


# -----------------------------
# MEDICAL Q&A (PROTECTED)
# -----------------------------
def generate_medical_answer(question: str) -> str:
    # Replace this with your real model later
    return (
        "I can help with that. Please share more details (age, symptoms, duration, test results) if relevant. "
        "If this is urgent (severe chest pain, trouble breathing, fainting, heavy bleeding), seek medical care immediately.\n\n"
        f"Your question: {question}"
    )


@app.route("/medical-qa", methods=["GET", "POST"])
def medical_qa():
    if not is_logged_in():
        flash("Please sign in to use Medical Q&A.", "error")
        return redirect(url_for("signin"))

    if "chat_history" not in session:
        session["chat_history"] = []

    # Reset chat
    if request.method == "POST" and "reset_chat" in request.form:
        session["chat_history"] = []
        return render_template("medical_qa.html", chat_history=session["chat_history"], user=session.get("user"))

    # New message
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            chat = session["chat_history"]
            chat.append({"role": "user", "content": question})
            answer = generate_medical_answer(question)
            chat.append({"role": "assistant", "content": answer})
            session["chat_history"] = chat

    return render_template("medical_qa.html", chat_history=session["chat_history"], user=session.get("user"))


if __name__ == "__main__":
    app.run(debug=True)
