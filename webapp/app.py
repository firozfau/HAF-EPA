from pathlib import Path
import json
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.predictor import recommend_top_employees_from_pdf

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"

app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    recommendations = []
    extracted_text = ""
    error = None
    uploaded_filename = ""

    if request.method == "POST":
        uploaded_file = request.files.get("project_pdf")

        if not uploaded_file or not uploaded_file.filename:
            error = "Please upload a PDF file."
        elif not uploaded_file.filename.lower().endswith(".pdf"):
            error = "Only PDF files are allowed."
        else:
            try:
                uploaded_filename = secure_filename(uploaded_file.filename)
                saved_path = UPLOAD_DIR / uploaded_filename
                uploaded_file.save(saved_path)

                recommendations, extracted_text = recommend_top_employees_from_pdf(
                    pdf_path=str(saved_path),
                    top_k=10,
                )

            except Exception as exc:
                error = str(exc)

    return render_template(
        "index.html",
        recommendations=recommendations,
        extracted_text=extracted_text,
        error=error,
        uploaded_filename=uploaded_filename,
    )


@app.route("/graph", methods=["POST"])
def graph():
    data = request.form.get("data")

    if data:
        recommendations = json.loads(data)
    else:
        recommendations = []

    return render_template("graph.html", recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)