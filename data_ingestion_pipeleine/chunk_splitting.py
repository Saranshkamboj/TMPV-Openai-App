import ast
import json


class Chunker:
    """
    Class for chunking text data into smaller chunks with a specified overlap and summarizing each chunk.
    """

    def __init__(self, chunk_size, openai_client, overlap_ratio=0.2):
        """
        Initialize the Chunker class.

        Args:
            chunk_size (int): The size of each chunk.
            overlap_ratio (float): The ratio of overlap between chunks. Default is 0.2 (20%).
        """
        print("INITIALIZING Chunker() ....")
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.openai_client = openai_client

    def chunk_text_with_overlap(self, texts):
        """
        Chunks the input text into larger chunks with overlap.

        Args:
            texts (list of str): The input text from each page.

        Returns:
            list: A list of text chunks.
        """

        combined_content = "\n".join(texts)
        chunk_size = len(combined_content) // len(texts)  # Approximate size of one page
        overlap_size = int(chunk_size * self.overlap_ratio)

        chunks = []
        for i in range(0, len(combined_content), chunk_size):
            chunk = combined_content[max(0, i - overlap_size) : i + chunk_size]
            chunks.append(chunk)
        return chunks

    def is_valid_json(self, json_string):
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False

    async def chunk_pages(self, extracted_data):
        """
        Chunks the text from each page of the extracted data with overlap and adds summarization metadata.

        Args:
            extracted_data (list): The list of dictionaries containing extracted text and HTML tables per page.

        Returns:
            list: A list of dictionaries with chunked and summarized text, HTML tables, and metadata per page.
        """
        chunked_data = []
        previous_chunk = ""

        for page_data in extracted_data:
            page_number = page_data["page_number"]
            page_text = page_data["text"]
            page_tables = page_data["tables"]

            if len(page_text) == 0:
                continue
            # Chunk the text from the current page with overlap
            text_chunks = self.chunk_text_with_overlap([page_text])

            for chunk in text_chunks:
                try:
                    # summarized_chunk = await self.summarize_chunk_with_gpt(chunk, previous_chunk, page_number)
                    summarized_chunk = {"text": page_text}
                    summarized_chunk["chunk_number"] = page_number
                    summarized_chunk["page_number"] = page_number
                    # Assign tables from the current page to the summarized chunk
                    summarized_chunk["tables"] = page_tables
                    chunked_data.append(summarized_chunk)
                    previous_chunk = chunk
                except Exception as e:
                    print(f"Error processing page {page_number} data: {e}")

        return chunked_data
