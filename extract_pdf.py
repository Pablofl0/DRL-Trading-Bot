import pdfplumber
import sys

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        return f"Error extracting text: {str(e)}"

if __name__ == "__main__":
    pdf_path = "doc_no_di.pdf"
    text = extract_text_from_pdf(pdf_path)
    
    # Save to txt file
    with open("pdf_content.txt", "w", encoding="utf-8") as f:
        f.write(text)
    
    # Print first 1000 characters to preview
    print(text[:1000] if len(text) > 1000 else text)
    print("\nFull content saved to pdf_content.txt") 