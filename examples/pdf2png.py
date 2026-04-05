import fitz  # PyMuPDF
import sys
import os

def convert_pdf_to_png(pdf_path):
    png_path = pdf_path.replace(".pdf", ".png")
    doc = fitz.open(pdf_path)
    # Get the first page
    page = doc[0]
    # Set higher resolution (300 DPI)
    # Default is 72 DPI, so zoom = 300 / 72
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
    pix.save(png_path)
    doc.close()
    print(f"Converted {pdf_path} to {png_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Make sure we're in the right directory relative to docs
    fig_dir = os.path.join(current_dir, "fig", "figure1")
    
    files = [
        "wine,weight_102_102_1_1000,W.pdf",
        "wine,weight_102_102_1_1000,X123.pdf",
        "wine,weight_102_102_1_1000,F.pdf"
    ]
    
    for f in files:
        pdf_path = os.path.join(fig_dir, f)
        if os.path.exists(pdf_path):
            convert_pdf_to_png(pdf_path)
        else:
            print(f"File not found: {pdf_path}", file=sys.stderr)
