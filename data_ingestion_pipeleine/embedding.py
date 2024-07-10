import asyncio
from azure.search.documents.indexes.models import (
    ComplexField,
    CorsOptions,
    SearchIndex,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    SearchField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    SemanticSearch,
)


class EmbeddingIndexer:
    """
    Class for creating embeddings from text chunks and indexing them to AI Search.
    """

    def __init__(
        self, search_index_client, search_client, openai_client, search_index_name
    ):
        """
        Initialize the EmbeddingIndexer class.

        Args:
            openai_api_key (str): The API key for OpenAI.
            search_service_endpoint (str): The endpoint of the Azure Search service.
            search_index_name (str): The name of the Azure Search index.
            search_api_key (str): The API key for the Azure Search service.
        """
        print("INITIALIZING EmbeddingIndexer() ....")
        self.openai_client = openai_client

        self.search_index_name = search_index_name
        self.search_index_client = search_index_client
        self.search_client = search_client

    async def create_index(self):
        """
        Create the search index with the specified schema.
        """
        try:
            print("CREATING INDEX ....")
            fields = [
                SimpleField(name="id", type=SearchFieldDataType.String, key=True),
                SimpleField(
                    name="page_number",
                    type=SearchFieldDataType.Int32,
                    sortable=True,
                    filterable=True,
                ),
                SearchableField(
                    name="text",
                    type=SearchFieldDataType.String,
                    filterable=False,
                    searchable=True,
                ),
                # ComplexField(name="metadata", type=SearchFieldDataType.Collection, fields=[
                #     SearchableField(name="chapter_title", type=SearchFieldDataType.String, searchable=True),
                #     SearchableField(name="section_title", type=SearchFieldDataType.String, searchable=True),
                #     SimpleField(name="summary", type=SearchFieldDataType.String, searchable=True)
                # ]),
                SimpleField(
                    name="file_name", type=SearchFieldDataType.String, sortable=True
                ),
                SimpleField(
                    name="chunk_number", type=SearchFieldDataType.Int32, sortable=True
                ),
                SimpleField(
                    name="tables",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.String),
                ),
                SimpleField(
                    name="model_name",
                    type=SearchFieldDataType.String,
                    sortable=True,
                    filterable=True,
                ),
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=1536,
                    vector_search_profile_name="myHnswProfile",
                ),
            ]

            # Configure the vector search configuration
            vector_search = VectorSearch(
                algorithms=[HnswAlgorithmConfiguration(name="myHnsw")],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                    )
                ],
            )

            semantic_config = SemanticConfiguration(
                name="my-semantic-config",
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[SemanticField(field_name="text")],
                    keywords_fields=[SemanticField(field_name="tables")],
                ),
            )

            # Create the semantic settings with the configuration
            semantic_search = SemanticSearch(configurations=[semantic_config])
            cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
            index = SearchIndex(
                name=self.search_index_name,
                fields=fields,
                vector_search=vector_search,
                cors_options=cors_options,
                semantic_search=semantic_search,
            )

            self.search_index_client.create_or_update_index(index=index)
            print(f"Index '{self.search_index_name}' created successfully.")
        except Exception as e:
            print(f"Error creating index: {e}")

    async def embed_chunk(self, chunk):
        """
        Embeds a text chunk using OpenAI.

        Args:
            chunk (str): The text chunk to embed.

        Returns:
            list: The embedding vector.
        """
        while True:
            try:
                print("EMBEDDING CHUNK ....")
                response = await self.openai_client.embeddings.create(
                    input=chunk, model="tmpv-dev-embedding"
                )

                return response.data[0].embedding
            except Exception as e:
                print(f"Error embedding chunk: {e}")
                pass

    async def index_chunk(self, embedding, document, file_name, model_name):
        """
        Indexes an embedded chunk to AI Search.

        Args:
            chunk (str): The text chunk.
            embedding (list): The embedding vector.
            metadata (dict): The metadata for the chunk.

        Returns:
            dict: The index response.
        """
        id_file_name = ""
        for char in file_name:
            if char not in [
                ".",
                " ",
                "-",
                "_",
                "@",
                "#",
                "!",
                "$",
                "%",
                "&",
                "*",
                "(",
                ")",
                "|",
            ]:
                id_file_name += char
        try:
            print("INDEXING CHUNK ....")

            document.update(
                {
                    "@search.action": "upload",
                    "id": f"{id_file_name}_{document['page_number']}_{document['chunk_number']}",
                    "file_name": file_name,
                    "model_name": model_name,
                    "embedding": embedding,
                }
            )

            result = self.search_client.upload_documents(documents=[document])

            return result
        except Exception as e:
            print(f"Error indexing chunk: {e}")
            return None

    async def embed_and_index_chunks(self, chunks_with_metadata, file_name, model_name):
        """
        Embeds and indexes multiple chunks with metadata.

        Args:
            chunks_with_metadata (list): A list of dictionaries containing chunks and their metadata.
        """
        try:
            print("EMBEDDING AND INDEX STARTED ....")
            tasks = []
            count = 0
            for item in chunks_with_metadata:
                chunk = item["text"]
                if len(chunk) <= 50:
                    print("*" * 90 + "\n\n", chunk)
                    continue

                chunk = file_name + "\n" + chunk
                print(
                    "Number of Documents: ",
                    len(chunks_with_metadata),
                    "      #       ",
                    count,
                )
                count += 1
                embedding = await self.embed_chunk(chunk)
                # output_filename = 'embedding.json'
                # with open(output_filename, 'a') as outfile:
                #     json.dump(embedding, outfile, indent=4)
                if embedding:
                    task = self.index_chunk(embedding, item, file_name, model_name)
                    tasks.append(task)
            await asyncio.gather(*tasks)
            # await self.openai_client.close()
            # await self.search_index_client.close()
            # await self.search_client.close()
        except Exception as e:
            print(f"Error in embedding and indexing chunks: {e}")
