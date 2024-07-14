import argparse
from pathlib import Path
from src.RAG import RAG
from src.pdf import DocumentAI, create_annotated_pdf
import os


def main(path_to_case_pdf: str, query: str, output_folder: str, *args, **kwargs):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    kwargs['layout_output_file_path'] = os.path.join(output_folder, kwargs['layout_output_file_name'])
    # generate layout aware text representation of the original document
    document_ai = DocumentAI()
    layout, elements, width, height = document_ai(Path(path_to_case_pdf), *args, **kwargs)  # below optional
    # build vector database and set up RAG pipeline
    vector_db = document_ai.build_vector_db(elements, *args, **kwargs)
    rag = RAG(vector_db, *args, **kwargs)
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
    default_llm_path = r'/Users/peter_zirui_wei/PycharmProjects/llama.cpp/models/mistral-7b-instruct-v0.1.Q4_K_M.gguf'
    parser.add_argument('--llm_path', type=str, default=default_llm_path)
    parser.add_argument('--output_folder', type=str, default=r'output')
    parser.add_argument('--layout_output_file_name', type=str, default=r'layout.txt')
    default_embedding_model = 'BAAI/bge-large-en-v1.5'
    parser.add_argument('--embedding_model', type=str, default=default_embedding_model)

    args = parser.parse_args()
    path_to_case_pdf = args.path_to_case_pdf
    query = args.query

    other_args = {k: v for k, v in vars(args).items() if k not in ['path_to_case_pdf', 'query']}

    main(path_to_case_pdf, query, **other_args)
