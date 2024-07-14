from io import BytesIO
from pathlib import Path
import dotenv
from PyPDF2 import PdfReader, PdfWriter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from reportlab.lib.colors import red
from reportlab.pdfgen import canvas
from unstructured.partition.pdf import partition_pdf

dotenv.load_dotenv()
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
import os

class DocumentAI:
    def __init__(self, embedding_model='BAAI/bge-large-en-v1.5'):
        self.embedding_model = embedding_model
        self.embedding_function = SentenceTransformerEmbeddings(model_name=self.embedding_model)

    def __call__(self, file_path: Path, *args, **kwargs):
        elements = partition_pdf(file_path, strategy="ocr_only")
        layout_result = generate_layout_string(elements, file_path, *args, **kwargs)
        print(layout_result)
        return (layout_result, elements, elements[0].metadata.coordinates.system.width,
                elements[0].metadata.coordinates.system.height)

    def build_vector_db(self, elements, persist_directory=r'data/vector_db_inpatient', *args, **kwargs):
        if os.path.exists(persist_directory):
            return Chroma(persist_directory=persist_directory, embedding_function=self.embedding_function)

        vectordb = Chroma(persist_directory=persist_directory, embedding_function=self.embedding_function)
        for element in elements:
            vectordb.add_documents([Document(
                page_content=element.text,
                metadata={
                    "page": element.metadata.page_number,
                    "coordinates": ','.join([str(x) for x in element.metadata.coordinates.points]),
                }
            )])
        return vectordb


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


def generate_layout_string(elements, file_path, layout_output_file_path='layout_output.txt', *args, **kwargs):
    # Group elements by page number
    pages = {}
    max_x, max_y = elements[0].metadata.coordinates.system.width, elements[0].metadata.coordinates.system.height
    reader = PdfReader(file_path)

    page = reader.pages[0]
    page_width = float(page.mediabox.width)
    page_height = float(page.mediabox.height)
    scale_x = page_width / max_x

    for element in elements:
        page_num = element.metadata.page_number
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(element)

    canvas_width = int(page_width)
    canvas_height = len(elements) + 3 * len(pages)  # Add space for page separators
    canvas = [[' ' for _ in range(canvas_width)] for _ in range(canvas_height)]
    y_offset = 0

    for page_num in sorted(pages.keys()):
        page_elements = pages[page_num]

        # Add page separator
        separator = f"--- Page {page_num} ---"
        start_x = (canvas_width - len(separator)) // 2
        for i, char in enumerate(separator):
            canvas[y_offset][start_x + i] = char
        y_offset += 1

        # Place each element on the canvas
        for element in page_elements:
            text = element.text
            x, y = element.metadata.coordinates.points[0]  # Using top-left corner as anchor point

            # Place each character of the text
            for i, char in enumerate(text):
                canvas_x = int(x * scale_x) + i
                canvas_y = y_offset
                if 0 <= canvas_x < canvas_width and 0 <= canvas_y < canvas_height:
                    canvas[canvas_y][canvas_x] = char

            y_offset += 1
        y_offset += 2
    # Convert the canvas to a string
    full_layout = '\n'.join(''.join(row).rstrip() for row in canvas).strip()

    # Write the full layout string to a file
    with open(layout_output_file_path, 'w', encoding='utf-8') as file:
        file.write(full_layout)

    return full_layout


if __name__ == '__main__':
    # Example Usage
    document_ai = DocumentAI()
    path = r'"data/inpatient_record.pdf"'
    # path = r'/Users/peter_zirui_wei/Downloads/inpatient_record-pages-1.pdf'
    document = document_ai(Path(path))
