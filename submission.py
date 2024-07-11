# import argparse
#
#
# def main():
#     """Write the entrypoint to your submission here"""
#     # TODO - import and execute your code here. Please put business logic into repo/src
#     raise NotImplementedError
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--path-to-case-pdf',
#                         metavar='path',
#                         type=str,
#                         help='Path to local test case with which to run your code')
#     args = parser.parse_args()
#     main()
# submission.py

import argparse
from pathlib import Path
from src.pdf import DocumentAI
from src.medical_record_processor import MedicalRecordProcessor


def main(path_to_case_pdf: str):
    document_ai = DocumentAI()

    documents = document_ai(Path(path_to_case_pdf))

    processor = MedicalRecordProcessor(documents)

    # 导出到 Excel
    processor.export_to_excel()

    # 生成 HTML 报告
    processor.generate_html_report(pdf_path=path_to_case_pdf)

    # 打印处理后的数据
    df = processor.create_dataframe()
    print(df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-to-case-pdf',
                        metavar='path',
                        type=str,
                        help='Path to local test case with which to run your code')
    args = parser.parse_args()
    main(args.path_to_case_pdf)