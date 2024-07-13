from pathlib import Path
import dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from unstructured.partition.pdf import partition_pdf

dotenv.load_dotenv()
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
import os


class DocumentAI:
    def __init__(self, embedding_model='BAAI/bge-large-en-v1.5'):
        self.embedding_model = embedding_model
        self.embedding_function = SentenceTransformerEmbeddings(model_name=self.embedding_model)

    def __call__(self, file_path: Path):
        elements = partition_pdf(file_path, strategy="ocr_only")
        layout_result = generate_layout_string(elements)
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


def generate_layout_string(elements, output_file_path='layout_output.txt'):
    # Group elements by page number
    pages = {}
    for element in elements:
        page_num = element.metadata.page_number
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(element)

    # Find the maximum coordinates across all pages
    max_x = max(max(max(coord[0] for coord in element.metadata.coordinates.points) for element in page) for page in
                pages.values())
    max_y = max(max(max(coord[1] for coord in element.metadata.coordinates.points) for element in page) for page in
                pages.values())

    # Create a canvas for the entire document
    scale_factor = 1  # Adjust this value to change the compactness
    canvas_width = int(max_x * scale_factor) + 1
    canvas_height = int(max_y * scale_factor) * len(pages) + len(pages)  # Add space for page separators
    canvas = [[' ' for _ in range(canvas_width)] for _ in range(canvas_height)]

    full_layout = ""
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
                canvas_x = int(x * scale_factor) + i
                canvas_y = int(y * scale_factor) + y_offset
                if 0 <= canvas_x < canvas_width and 0 <= canvas_y < canvas_height:
                    canvas[canvas_y][canvas_x] = char

        # Update y_offset for the next page
        y_offset += int(max_y * scale_factor) + 1

    # Convert the canvas to a string
    full_layout = '\n'.join(''.join(row).rstrip() for row in canvas).strip()

    # Write the full layout string to a file
    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(full_layout)

    return full_layout


if __name__ == '__main__':
    # Example Usage
    document_ai = DocumentAI()
    path = r'"data/inpatient_record.pdf"'
    # path = r'/Users/peter_zirui_wei/Downloads/inpatient_record-pages-1.pdf'
    document = document_ai(Path(path))
