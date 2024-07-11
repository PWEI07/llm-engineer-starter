import mimetypes
import os
from pathlib import Path
import PyPDF2
import tempfile
import dotenv

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud.documentai_v1 import Document

# Load Env Files.
# This will return True if your env vars are loaded successfully
dotenv.load_dotenv()

class DocumentAI:
    # def __init__(self) -> None:
    #     # Set endpoint to EU
    #     options = ClientOptions(api_endpoint="eu-documentai.googleapis.com:443")
    #
    #     # Instantiate a client
    #     self.client = documentai.DocumentProcessorServiceClient(client_options=options)
    #
    #     # Set up the processor name
    #     project_id = os.getenv("GCP_PROJECT_ID")
    #     location = "eu"  # Assuming EU location based on the endpoint
    #     processor_id = os.getenv("GCP_PROCESSOR_ID")
    #     self.processor_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

    """Wrapper class around GCP's DocumentAI API."""
    def __init__(self) -> None:

        self.client_options = ClientOptions(  # type: ignore
            api_endpoint=f"{os.getenv('GCP_REGION')}-documentai.googleapis.com",
            credentials_file=os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        )
        self.client = documentai.DocumentProcessorServiceClient(client_options=self.client_options)
        self.processor_name = self.client.processor_path(
            os.getenv("GCP_PROJECT_ID"),
            os.getenv("GCP_REGION"),
            os.getenv("GCP_PROCESSOR_ID")
        )

    def split_pdf(self, file_path: Path, max_pages: int = 15):
        temp_dir = tempfile.mkdtemp()
        split_pdfs = []

        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)

            for start in range(0, total_pages, max_pages):
                end = min(start + max_pages, total_pages)
                output = PyPDF2.PdfWriter()

                for page in range(start, end):
                    output.add_page(pdf_reader.pages[page])

                split_file = Path(temp_dir) / f"split_{start + 1}_{end}.pdf"
                with open(split_file, "wb") as output_file:
                    output.write(output_file)

                split_pdfs.append(split_file)

        return split_pdfs
    def process_document(self, file_path: Path) -> documentai.Document:
        with open(file_path, "rb") as file:
            content = file.read()

        mime_type = mimetypes.guess_type(file_path)[0]
        raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
        # document = {"content": content, "mime_type": mime_type}
        request = documentai.ProcessRequest(
            name=self.processor_name,
            raw_document=raw_document
        )
        result = self.client.process_document(request=request)
        return result.document

    def __call__(self, file_path: Path):
        split_pdfs = self.split_pdf(file_path)
        processed_documents = []

        for pdf in split_pdfs:
            try:
                processed_document = self.process_document(pdf)
                processed_documents.append(processed_document)
            except Exception as e:
                print(f"Error processing {pdf}: {str(e)}")

        # Clean up temporary files
        for pdf in split_pdfs:
            pdf.unlink()
        os.rmdir(os.path.dirname(split_pdfs[0]))

        return processed_documents

    # def __call__(self, file_path: Path) -> Document:
    #     """Convert a local PDF into a GCP document. Performs full OCR extraction and layout parsing."""
    #
    #     # Read the file into memory
    #     with open(file_path, "rb") as file:
    #         content = file.read()
    #
    #     mime_type = mimetypes.guess_type(file_path)[0]
    #     raw_document = documentai.RawDocument(content=content, mime_type=mime_type)
    #
    #     # Configure the process request
    #     request = documentai.ProcessRequest(
    #         name=self.processor_name,
    #         raw_document=raw_document
    #     )
    #
    #     result = self.client.process_document(request=request)
    #     document = result.document
    #
    #     return document


if __name__ == '__main__':

    # Example Usage
    document_ai = DocumentAI()
    document = document_ai(Path("data/inpatient_record.pdf"))
