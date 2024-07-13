import argparse
from pathlib import Path
from src.RAG import RAG
from src.pdf_unstructured import DocumentAI
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.colors import red
from io import BytesIO
import os


def create_annotated_pdf(path_to_case_pdf, query, source, output_folder, width, height):
    reader = PdfReader(path_to_case_pdf)

    page_number = source.metadata['page']
    page = reader.pages[page_number - 1]  # PDF 页面索引从 0 开始
    page_width = float(page.mediabox.width)
    page_height = float(page.mediabox.height)
    scale_x = page_width / width
    scale_y = page_height / height

    output = PdfWriter()
    output.add_page(page)

    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=(page_width, page_height))
    can.setStrokeColor(red)
    can.setFillColor(red)

    coordinates = eval(source.metadata['coordinates'])
    x1, y1 = coordinates[0]
    x2, y2 = coordinates[2]

    x1_scaled = x1 * scale_x
    y1_scaled = y1 * scale_y
    x2_scaled = x2 * scale_x
    y2_scaled = y2 * scale_y

    can.rect(x1_scaled, page_height - y2_scaled, x2_scaled - x1_scaled, y2_scaled - y1_scaled, stroke=1, fill=0)

    can.setFont("Helvetica", 12)
    can.drawString(x1_scaled, page_height - y1_scaled - 32, f"Question: {query}")

    can.save()

    packet.seek(0)
    new_pdf = PdfReader(packet)
    page.merge_page(new_pdf.pages[0])

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f"{query}.pdf")
    with open(output_path, "wb") as output_file:
        output.write(output_file)

    print(f"Annotated PDF saved as: {output_path}")

    import subprocess
    subprocess.Popen([output_path], shell=True)


def main(path_to_case_pdf: str, query: str, *args, **kwargs):
    document_ai = DocumentAI()
    layout, elements, width, height = document_ai(Path(path_to_case_pdf))
    vector_db = document_ai.build_vector_db(elements, *args, **kwargs)
    rag = RAG(vector_db)
    answer = rag.query(query)
    print(f"query:\n{query}\n\nanswer:\n{answer['result']}")
    output_folder = "output"
    create_annotated_pdf(path_to_case_pdf, query, answer['source_documents'][0], output_folder, width, height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-case-pdf',
                        metavar='path',
                        type=str,
                        required=True,
                        help='Path to local test case with which to run your code')
    default_query = "what is the medical history of the patient's shoulder, when did it happen?"
    parser.add_argument('--query', type=str, default=default_query)
    parser.add_argument('--persist_directory', type=str, default=r'data/vector_db_inpatient')

    args = parser.parse_args()
    path_to_case_pdf = args.path_to_case_pdf
    query = args.query

    other_args = {k: v for k, v in vars(args).items() if k not in ['path_to_case_pdf', 'query']}

    main(path_to_case_pdf, query, **other_args)
