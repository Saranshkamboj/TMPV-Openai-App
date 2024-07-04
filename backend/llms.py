from prompts import PromptSet
from openai import AzureOpenAI


class LLMRsponseGeneration:
    def __init__(
        self, search_client, openai_client, search_index_client, search_index_name
    ):
        self.search_index_client = search_index_client
        self.search_index_name = search_index_name
        self.search_client = search_client
        self.openai_client = openai_client

    def get_embeddings(self, text: str):
        client = AzureOpenAI(
            azure_endpoint="https://gpt-demo-openai.openai.azure.com/",
            api_key="17e88c37e0d845de9fd65d763b3a522f",
            api_version="2023-03-15-preview",
        )
        embedding = client.embeddings.create(
            input=[text], model="text-embedding-ada-002"
        )

        return embedding.data[0].embedding

    async def generate_responses(self, user_query, message):

        response = await self.openai_client.chat.completions.create(
            model="tmpv-cx-gpt-model",
            messages=message,
            n=1,
            stop=None,
            temperature=0.1,
        )

        return response.choices[0].message.content
