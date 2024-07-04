from azure.search.documents.models import VectorizedQuery
import tiktoken
import time


class ContextRetrieval:

    def __init__(self, search_client, openai_client):
        self.search_client = search_client
        self.openai_client = openai_client

    # test function
    async def get_embeddings(self, text: str):

        embedding = await self.openai_client.embeddings.create(
            input=[text], model="tmpv-dev-embedding"
        ) 

        return embedding.data[0].embedding

    def count_tokens(paragraph, model="gpt2"):
        # Load the appropriate tokenizer
        enc = tiktoken.get_encoding(model)

        # Encode the paragraph to get the tokens
        tokens = enc.encode(paragraph)

        # Return the number of tokens
        return len(tokens)

    async def retrieve(self, query, model_name, knn=10):

        if model_name == "Others":
            filter_query = None
        else:
            filter_query = f"model_name eq '{model_name}'"
        print("filter query: ", filter_query)
        vector_query = VectorizedQuery(
            vector=await self.get_embeddings(query),
            k_nearest_neighbors=knn,
            fields="embedding",
        )

        result = self.search_client.search(
            vector_queries=[vector_query],
            filter=filter_query,
        )

        formated_context = []
        context = ""
        token_count = 0
        for item in result:
            context += f"Model Name: {item['model_name']}\nPage Number: {item['page_number']}\nText: {item['text']}\n\n"
            formated_context.append(
                {
                    "search_score": item["@search.score"],
                    "page_no": item["page_number"],
                    "content": item["text"],
                }
            )
            token_count += ContextRetrieval.count_tokens(context)

        context = context.replace("\\n", "")
        return formated_context, context
