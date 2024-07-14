# Medical Document AI Analyzer

## Abstract

This project presents an advanced system for analyzing medical documents using state-of-the-art AI technologies. It combines OCR capabilities, layout-aware text representation, and a Retrieval-Augmented Generation (RAG) pipeline to efficiently process and query medical records. Key features include:

- Layout-preserving text extraction from PDF documents
- Advanced RAG pipeline with embedding redundancy filtering and reranking
- Visual highlighting of relevant information in generated PDFs
- Utilization of cutting-edge open-source embedding models and LLMs, delivering fast and robust answers to user queries based on medical records

This solution offers medical professionals a powerful tool to quickly extract information from raw files and accurately answer questions based on the information in the file, potentially improving decision-making and patient care.

## Table of Contents

1. [Technology Stack](#technology-stack)
2. [Mechanism](#mechanism)
3. [Usage](#usage)
4. [Future Research](#future-research)

## Technology Stack

### Unstructured.io

We use Unstructured.io for OCR and text extraction. Unstructured provides open-source components for ingesting and pre-processing various types of unstructured data, including images and text documents (such as PDFs, HTML, Word docs, and more). These tools streamline and optimize data processing workflows for large language models (LLMs) by transforming unstructured data into structured outputs. It has an active community, regular updates, and robust OCR capabilities for various document types.

For more information, visit [Unstructured.io](https://unstructured.io/).

### BGE-large-en Embedding

Our system utilizes the [BGE-large-en embedding model](https://huggingface.co/BAAI/bge-large-en) for text embedding. This model offers several advantages:

- State-of-the-art performance on semantic search tasks
- Optimization for English language processing
- Versatility: BGE can map any text to a low-dimensional dense vector, making it suitable for tasks like retrieval, classification, clustering, and semantic search
- Large-Scale Training: Pre-trained using RetroMAE and trained on large-scale pair data using contrastive learning

### Mistral-7B

For language modeling, we employ Mistral-7B, a cutting-edge language model that offers:
- Accuracy: Mistral-7B is a 7.3 billion parameter language model that outperforms Llama 2 13B on all benchmarks. Its impressive accuracy makes it an attractive choice for various NLP tasks.
- Robustness: The model handles out-of-vocabulary (OOV) words effectively, ensuring robust performance across different text inputs.
- Grouped-Query Attention (GQA): Mistral-7B leverages GQA for faster inference, enhancing its efficiency while maintaining performance.
- Sliding Window Attention (SWA): SWA allows Mistral-7B to handle longer sequences with reduced computational cost, making it versatile for various use cases.

## Mechanism

1. **OCR and Layout Extraction**: Using Unstructured.io, we extract text and layout information from PDF documents, preserving spatial relationships.

2. **Text Embedding**: The extracted text is embedded using the BGE-large-en model, creating a semantic representation of the content.

3. **Vector Storage**: Embeddings are stored in a lightweight vector database, Chroma, for efficient retrieval.

4. **RAG Pipeline**: We implement a Retrieval-Augmented Generation pipeline that includes:
   - Embedding redundancy filter to eliminate redundant information in first-stage retrieved results. See 
     [EmbeddingsRedundantFilter](https://api.python.langchain.com/en/latest/document_transformers/langchain_community.document_transformers.embeddings_redundant_filter.EmbeddingsRedundantFilter.html) for details.
   - Reranking of retrieved information to improve relevancy ranking, placing the most relevant information at the beginning of the prompt fed into the LLM. This addresses the LLM's tendency to utilize information at the start better than information in the middle (known as the "lost in the middle" problem).
   - Integration with the Mistral-7B language model for generating responses.

5. **Visual Feedback**: The system generates a new PDF highlighting relevant text in red bounding boxes in an output folder based on user queries, facilitating easy review of source information.

## Usage

### Configuration

The demo runs on a MacBook with an M1 chip. To use the system:

1. Run the `main()` function in `submission.py`.
2. Provide the `path_to_case_pdf` parameter to specify the location of the PDF to be analyzed.

Additional important parameters:

- `llm_path`: Path to the GGUF quantized LLM model.

**GPT-Generated Unified Format (GGUF)** is a quantized file format specifically designed for LLMs. It offers several advantages, including faster inference speed, particularly for Mac and other consumer-grade hardware. 

- `output_folder`: The path to the output folder to store the layout-aware text file and the PDF with red bounding boxes around relevant text for the user's question.

## Future Research

1. **Temporal Data Association**: 
   - Develop methods to reliably associate dates with each medical event. By ensuring there's a corresponding date for each medical event we retrieve, we can use the date as metadata when storing them in the vector database and prompt the LLM to prioritize the latest information when answering questions. In medical records, patient conditions change over time, so the same test result may show different evaluations at different times. When answering a question, we should resort to the most recent information as facts.
   - Utilize GCP Document AI's parent-child field relationships to ensure date inclusion for each medical event.

2. **Fine-tuning OCR Models**: 
   - Create human-annotated datasets of medical documents.
   - Fine-tune pre-trained OCR models to improve accuracy on medical-specific layouts. We can manually label some documents and use them as data to "teach" the pre-trained model to better adapt to our desired extraction outcomes.

3. **Type-Aware Information Filtering**: 
   - Implement intelligent filtering based on element types (e.g., headers vs. body text). We can take into account the element type identified in OCR to better distinguish between medical event-related information and non-relevant data. For example, in the sample PDF, "4/2/2024 1:22:35 PM EDT..." appeared multiple times in the document header, but it's more likely an indication of when the document was faxed, rather than the exact time a medical event occurred.

4. **Enhanced RAG Techniques**: 
   - Experiment with hybrid retrieval methods combining dense and sparse representations.
   - Implement iterative retrieval techniques to improve answer accuracy.

By addressing these areas, we aim to create an even more robust and accurate system for medical document analysis, potentially revolutionizing how healthcare professionals interact with patient records.