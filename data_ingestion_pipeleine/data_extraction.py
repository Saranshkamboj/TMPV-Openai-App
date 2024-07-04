import html
import fitz
import PyPDF2


class DocumentExtractor:
    """
    Class for extracting text and HTML tables from PDFs using Azure Document Intelligence.
    """

    def __init__(self, document_analysis_client):
        print("INITIALIZING DocumentExtractor() ....")
        self.client = document_analysis_client

    def extract_text_from_pdf(document, page_num):
        try:
            page = document.load_page(page_num)
            text = page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text: {e}")
            return None

    def count_pdf_pages(file_path):
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            return len(reader.pages)

    def chunk_overlapping(data):
        prev_chunk = ""
        for i in range(len(data)):
            current_chunk = data[i]["text"]
            if i == len(data) - 1:
                next_chunk = ""
            else:
                next_chunk = data[i + 1]["text"]

            print("*" * 80, "\n\n")
            print(prev_chunk)
            print("*" * 80, "\n\n")
            print(current_chunk)
            print("*" * 80, "\n\n")
            print(next_chunk)
            prev_chunk_words = prev_chunk.split(" ")
            next_chunk_words = next_chunk.split(" ")
            completed_chunk = (
                " ".join(prev_chunk_words[-int(len(prev_chunk_words) * 0.2) :])
                + "\n"
                + current_chunk
                + "\n"
                + " ".join(next_chunk_words[: int(len(next_chunk_words) * 0.2)])
            )
            data[i]["text"] = completed_chunk
            prev_chunk = current_chunk
            print("*" * 80, "\n\n")
            print(completed_chunk)
        return data

    async def extract_data_from_url(self, document_url, file_path):
        """
        Extracts text and HTML tables from a PDF.

        Args:
            document_url (str): URL to the PDF file.

        Returns:
            list: A list of dictionaries containing extracted text and HTML tables per page.
        """
        extracted_data = []
        try:
            document_length = DocumentExtractor.count_pdf_pages(file_path)

            for i in range(document_length // 2000 + 1):
                start = 2000 * i + 1
                end = 2000 * (i + 1)
                poller = await self.client.begin_analyze_document_from_url(
                    model_id="prebuilt-layout",
                    document_url=document_url,
                    pages=f"{start-end}",
                )
                result = await poller.result()

                for page_num, page in enumerate(result.pages):
                    page_data = {
                        "page_number": str(page_num + 1),  # Convert to 1-based indexing
                        "text": "",
                        "tables": [],
                    }

                    page_data["text"] = " ".join([t.content for t in page.lines])

                    if len(page_data["text"]) < 500:
                        print(len(page_data["text"]))
                        # print(page_data['text'])
                    # Process tables and convert to HTML
                    if result.tables:
                        for table_idx, table in enumerate(result.tables):
                            try:
                                if (
                                    table.cells[0].bounding_regions[0].page_number
                                    == page_num + 1
                                ):
                                    pass
                                else:
                                    continue
                            except Exception as e:
                                continue
                            table_html = ""
                            if table.cells:
                                # Extract table data
                                table_rows = self.get_table_rows(table)
                                table_html = self.table_to_html(table_rows)

                            page_data["tables"].append(table_html)

                    extracted_data.append(page_data)
            extracted_data = DocumentExtractor.chunk_overlapping(extracted_data)

        except Exception as e:
            print(f"Error using Azure Document Intelligence: {e}")

        # await self.client.close()
        return extracted_data

    def get_table_rows(self, table) -> list[list[str]]:
        """
        Extracts table data (text content of each cell) into a list of lists.

        Args:
            table (DocumentTable): The DocumentTable object containing table information.

        Returns:
            list[list[str]]: A list of lists, where each inner list represents a row with cell contents.
        """
        table_rows = []
        for row_index in range(table.row_count):
            row_cells = []
            for cell in table.cells:
                if cell.row_index == row_index:
                    row_cells.append(cell.content)
            table_rows.append(row_cells)
        return table_rows

    def table_to_html(self, table_rows: list[list[str]]) -> str:
        """
        Converts a list of lists representing table data to HTML.

        Args:
            table_rows (list[list[str]]): A list of lists containing cell contents for each row.

        Returns:
            str: The HTML representation of the table.
        """
        table_html = "<table>"
        for row in table_rows:
            table_html += "<tr>"
            for cell in row:
                table_html += f"<td>{html.escape(cell)}</td>"
            table_html += "</tr>"
        table_html += "</table>"
        return table_html


# Load environment variables from a .env file
# load_dotenv()

# Set standard output to use UTF-8 encoding
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Set up your Azure Document Intelligence endpoint
# DOCUMENTINTELLIGENCE_ENDPOINT = os.environ.get("DOCUMENTINTELLIGENCE_ENDPOINT")
# DOCUMENTINTELLIGENCE_API_KEY = os.environ.get("DOCUMENTINTELLIGENCE_API_KEY")

# Example URL of the PDF
# pdf_url = "https://scadlsmvpsea01.blob.core.windows.net/data/UWC Berhad - Section 4 (without price).pdf"

# async def main():
#     # Instantiate the DocumentExtractor with the endpoint
#     extractor = DocumentExtractor(DOCUMENTINTELLIGENCE_ENDPOINT, DOCUMENTINTELLIGENCE_API_KEY)

#     # Extract data from the PDF URL
#     extracted_data = await extractor.extract_data_from_url(pdf_url)

#     output_filename = 'extracted_data.json'
#     with open(output_filename, 'w') as outfile:
#         json.dump(extracted_data, outfile, indent=4)

# # Run the main function
# asyncio.run(main())
