from prompts import PromptSet
from backend.llms import LLMRsponseGeneration
from utils.blob import connect_blob
import json


class QueryAnalyzer:

    def __init__(self):
        pass

    async def intents_extraction(task_trigger, query):
        prompt = PromptSet.intent_extraction
        response = await task_trigger.generate_responses(query, prompt)

        return response

    async def query_rephraser(messages, task_trigger, query):
        prompt = PromptSet.query_rephraser_prompt

        chat_history = []
        for i in range(len(messages)):
            chat_history.append(
                {
                    "role": "user",
                    "content": str({"extracted_search_term": messages[i]["query"]}),
                }
            )
            chat_history.append(
                {"role": "assistant", "content": messages[i]["rephrased_query"]}
            )

        context = [{"role": "system", "content": prompt}]
        context = context + chat_history + [{"role": "user", "content": query}]

        try:
            response = await task_trigger.generate_responses(query, context)
            response = json.loads(response)
            response = response["extracted_search_term"]
            if len(response) == 0:
                response = query
        except Exception as e:
            response = query

        return response
