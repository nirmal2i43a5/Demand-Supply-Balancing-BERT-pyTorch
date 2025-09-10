import io
from PyPDF2 import PdfReader

def extract_text_from_pdf_bytes(pdf_bytes) :
    
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    # print(pages)
    
    return "\n".join(pages).strip()
