import pandas as pd
from collections import defaultdict
from datetime import datetime
import re
import hashlib
import os

START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 4, 2)

def parse_date(date_str):
    return datetime.strptime(date_str, '%m/%d/%Y')
def is_valid_date(date_str):
    date = parse_date(date_str)
    return datetime(2020, 1, 1) < date < datetime(2024, 4, 2)
class MedicalRecordProcessor:
    def __init__(self, documents):
        self.documents = documents
        self.text = " ".join(doc.text for doc in documents)
        self.pages = []
        self.page_offset = 0
        for doc in documents:
            for page in doc.pages:
                self.pages.append((self.page_offset + page.page_number, page))
            self.page_offset += len(doc.pages)

    def process(self):
        events = self.extract_events()
        return self.organize_events(events)

    def extract_events(self):
        events = []
        for page_number, page in self.pages:
            try:
                page_text = self.get_page_text(page)
                date_matches = list(re.finditer(r'\b(\d{1,2}/\d{1,2}/\d{4})\b', page_text))
                for match in date_matches:
                    date = match.group(1)
                    start_pos = match.start()
                    event_text = self.extract_event_text(page_text, start_pos)
                    events.append({
                        'date': date,
                        'event': event_text,
                        'page': page_number,
                        'position': start_pos
                    })
            except Exception as e:
                print(f"Error processing page {page_number}: {str(e)}")
        return events

    def get_page_text(self, page):
        text = ""
        for segment in page.layout.text_anchor.text_segments:
            start_index = getattr(segment, 'start_index', 0)
            end_index = segment.end_index
            text += self.text[start_index:end_index]
        return text

    def extract_event_text(self, text, start_index, max_length=200):
        end_index = min(start_index + max_length, len(text))
        return text[start_index:end_index].replace('\n', ' ').strip()

    def organize_events(self, events):
        organized_events = defaultdict(lambda: defaultdict(list))
        for event in events:
            date_obj = datetime.strptime(event['date'], '%m/%d/%Y')
            event_hash = hashlib.md5(event['event'].encode()).hexdigest()
            organized_events[date_obj][event_hash].append(event)

        result = []
        for date, event_dict in sorted(organized_events.items()):
            for event_hash, event_list in event_dict.items():
                result.append({
                    'date': date,
                    'event': event_list[0]['event'],
                    'page': event_list[0]['page'],
                    'position': event_list[0]['position'],
                    'duplicates': len(event_list) - 1,
                    'duplicate_locations': [f"Page {e['page']}, Pos {e['position']}" for e in event_list[1:]]
                })

        return result

    def create_dataframe(self):
        events = self.process()
        df = pd.DataFrame(events)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        return df

    def export_to_excel(self, filename='medical_records.xlsx'):
        df = self.create_dataframe()
        df.to_excel(filename, index=False)
        print(f"Data exported to {filename}")

    def generate_html_report(self, filename='medical_records_report.html', pdf_path=None):
        df = self.create_dataframe()
        pdf_link = f'<a href="file://{os.path.abspath(pdf_path)}">Original PDF</a>' if pdf_path else ''

        table_rows = []
        for _, row in df.iterrows():
            table_rows.append(f"""
            <tr>
                <td>{row['date'].strftime('%Y-%m-%d')}</td>
                <td>{row['event']}</td>
                <td><a href="#" onclick="jumpToPDF({row['page']}, {row['position']})">Page {row['page']}, Pos {row['position']}</a></td>
                <td>{row['duplicates']} {', '.join(row['duplicate_locations']) if row['duplicates'] > 0 else ''}</td>
            </tr>
            """)

        table_content = ''.join(table_rows)

        html_content = f"""
        <html>
        <head>
            <title>Medical Records Report</title>
            <style>
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    border: 1px solid black;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
            </style>
            <script>
                function jumpToPDF(page, position) {{
                    alert(`Jumping to page ${{page}}, position ${{position}}`);
                }}
            </script>
        </head>
        <body>
            <h1>Medical Records Report</h1>
            {pdf_link}
            <table>
                <tr>
                    <th>Date</th>
                    <th>Event</th>
                    <th>Location</th>
                    <th>Duplicates</th>
                </tr>
                {table_content}
            </table>
        </body>
        </html>
        """

        with open(filename, 'w') as f:
            f.write(html_content)
        print(f"HTML report generated: {filename}")