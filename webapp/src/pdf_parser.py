import fitz

# Extract text content from a PDF file using PyMuPDF.

# 1. Open the PDF document from the given file path.
# 2. Iterate through all pages and extract text content.
# 3. Collect non-empty text from each page.

# 4. Combine all page texts into a single string.

# 5. Raise an error if no text could be extracted.

# 6. Return the cleaned full text.

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    pages = []

    for page in doc:
        text = page.get_text("text")
        if text:
            pages.append(text)

    full_text = "\n".join(pages).strip()

    if not full_text:
        raise ValueError("Could not extract text from the uploaded PDF.")

    return full_text