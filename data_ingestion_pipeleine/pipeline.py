import os
from .blob_storage import BlobStorage
from .data_extraction import DocumentExtractor
from .chunk_splitting import Chunker
from .embedding import EmbeddingIndexer


class Pipeline:
    def __init__(self):
        self.blob_service_client = None
        self.document_extractor = None
        self.chunker = None
        self.embedding_indexer = None

    async def initialize_components(
        self,
        blob_service_client,
        openai_client,
        document_analysis_client,
        search_index_client,
        search_client,
        search_index_name,
        model_name,
    ):
        try:
            self.blob_storage = BlobStorage(blob_storage_client=blob_service_client)
            self.document_extractor = DocumentExtractor(
                document_analysis_client=document_analysis_client
            )
            self.chunker = Chunker(chunk_size=1536, openai_client=openai_client)
            self.embedding_indexer = EmbeddingIndexer(
                search_client=search_client,
                search_index_client=search_index_client,
                openai_client=openai_client,
                search_index_name=search_index_name,
            )
            self.model_name = model_name
        except Exception as e:
            print(f"Error initializing components: {e}")

    async def run_pipeline(self, file_path, file_name, search_index_name):
        try:
            # dummy function
            def start():
                return "start"

            file_url = ""

            count = 0
            for function in [
                start,
                self.blob_storage.upload_to_blob_storage,
                self.document_extractor.extract_data_from_url,
                self.chunker.chunk_pages,
                self.embedding_indexer.create_index,
                self.embedding_indexer.embed_and_index_chunks,
            ]:

                if count == 0:
                    res = function()
                    response_to_send = "Uploading file..."
                elif count == 1:
                    print("UPLOADING FILE ...")
                    file_url = await function(file_path)
                    response_to_send = "Extracting data from file..."
                elif count == 2:
                    print("EXTRACTING DATA FROM FILE ...")
                    extracted_data = await function(file_url, file_path)
                    response_to_send = "Chunking extracted data..."
                elif count == 3:
                    print("CHUNKING EXTRACTED DATA ...")
                    chunked_data = await function(extracted_data)
                    response_to_send = "Creating search index..."
                elif count == 4:
                    await function()
                    response_to_send = "Embedding and indexing data..."
                elif count == 5:
                    print("EMBEDDING AND INDEXING DATA ...")
                    await function(chunked_data, file_name, self.model_name)
                    response_to_send = (
                        f"File '{file_name}' uploaded and processed successfully."
                    )
                    # os.remove(file_path)
                count += 1

                output = {
                    "success": True,
                    "file_name": file_name,
                    "file_url": file_url,
                    "search_index_name": search_index_name,
                    "response": response_to_send,
                }
                yield output

        except Exception as e:
            raise e
            print(f"Error running pipeline: {e}")


# async def main():
#     pipeline = Pipeline()
#     await pipeline.initialize_components()
#     file_path = "data_ingestion_pipeleine\ATech - Prospectus (Part 1)_Final.pdf"
#     await pipeline.run_pipeline(file_path)

# if __name__ == "__main__":
#     asyncio.run(main())
