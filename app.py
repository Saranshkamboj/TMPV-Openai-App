import os
import time
import re
from datetime import datetime
import json
import uvicorn
from fastapi import FastAPI, Form, Request, Response


from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from typing import List
from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from azure.identity.aio import DefaultAzureCredential
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob.aio import BlobServiceClient
from openai import AsyncAzureOpenAI
from azure.ai.formrecognizer.aio import DocumentAnalysisClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from utils.query_analyzer import QueryAnalyzer
from backend.retrieve import ContextRetrieval
from prompts import PromptSet
from utils.blob import connect_blob


from data_ingestion_pipeleine.pipeline import Pipeline
from backend.llms import LLMRsponseGeneration
from data_ingestion_pipeleine.blob_storage import BlobStorage


BLOB_STORAGE_ACCOUNT_NAME = os.getenv("BLOB_STORAGE_ACCOUNT_NAME")
BLOB_STORAGE_ACCOUNT_KEY = os.getenv("BLOB_STORAGE_ACCOUNT_KEY")

DOCUMENTINTELLIGENCE_ENDPOINT = os.getenv("DOCUMENTINTELLIGENCE_ENDPOINT")
DOCUMENTINTELLIGENCE_API_KEY = os.getenv("DOCUMENTINTELLIGENCE_API_KEY")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")

AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")


blob_service_client = BlobServiceClient(
    account_url=f"https://{BLOB_STORAGE_ACCOUNT_NAME}.blob.core.windows.net/",
    credential=(
        DefaultAzureCredential()
        if not BLOB_STORAGE_ACCOUNT_KEY
        else BLOB_STORAGE_ACCOUNT_KEY
    ),
)

openai_client = AsyncAzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)


document_analysis_client = DocumentAnalysisClient(
    endpoint=DOCUMENTINTELLIGENCE_ENDPOINT,
    credential=(
        DefaultAzureCredential()
        if not DOCUMENTINTELLIGENCE_API_KEY
        else AzureKeyCredential(DOCUMENTINTELLIGENCE_API_KEY)
    ),
)


search_index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)

search_client = SearchClient(
    endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)


# Creating a FastAPI object
app = FastAPI()

pipeline = Pipeline()


async def upload_file(file, contents, model_name):
    if not file:
        return {"success": False, "error": "No file was uploaded."}

    try:
        # Get file metadata
        file_name = file.filename
        search_index_filename = re.sub(r"[^\w-]", "", file_name)
        search_index_filename = search_index_filename.lower()
        folder_path = "static"

        search_index_name = AZURE_SEARCH_INDEX_NAME

        # Check if the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, file_name)

        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        search_client = SearchClient(
            endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
            index_name=search_index_name,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
        )

        await pipeline.initialize_components(
            blob_service_client,
            openai_client,
            document_analysis_client,
            search_index_client,
            search_client,
            search_index_name,
            model_name,
        )

        bs = BlobStorage(blob_service_client)

        async for final_output in pipeline.run_pipeline(
            file_path, file_name, search_index_name
        ):
            final_output.update({"timestamp": datetime.utcnow().isoformat()})
            await bs.upload_json(final_output, f"{file_name}-uf.json")

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/uploadfile")
async def submit_form(
    background_tasks: BackgroundTasks, file: UploadFile, model_name: str = Form()
):
    file_name = file.filename
    contents = await file.read()
    file_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{file_name}"
    background_tasks.add_task(upload_file, file, contents, model_name)
    x = BlobStorage(blob_service_client)
    return x.generate_sas_url(f"{file_name}-uf.json") 


@app.post("/vehicle-insights")
async def vehicle_insights(query: str = Form(), chat_session_id: str = Form()):

    search_index_name = AZURE_SEARCH_INDEX_NAME

    try:
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_SERVICE_ENDPOINT,
            index_name=search_index_name,
            credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
        )

        task_trigger = LLMRsponseGeneration(
            search_client, openai_client, search_index_client, search_index_name
        )
        bs = BlobStorage(blob_service_client)
        # intents = await QueryAnalyzer.intents_extraction(task_trigger, query)
        search_term = await QueryAnalyzer.query_rephraser(
            chat_session_id, task_trigger, query
        )
        print("\n\n", search_term, "\n\n")
        cr = ContextRetrieval(search_client, openai_client)
        context = await cr.retrieve(search_term, 30)
        print(context)
        prompt = PromptSet.doc_specific_prompt
        prompt = prompt.replace("{context}", context)

        if await connect_blob(chat_session_id, "blob_status"):
            chat_history = await connect_blob(chat_session_id, "download")
            context = (
                [{"role": "system", "content": prompt}]
                + chat_history
                + [{"role": "user", "content": query}]
            )
        else:
            context = [{"role": "system", "content": prompt}] + [
                {"role": "user", "content": query}
            ]

        response = await task_trigger.generate_responses(query, context)
        context_to_store = context[1:] + [{"role": "assistant", "content": response}]
        await connect_blob(chat_session_id, "upload", context_to_store)

        return {"success": True, "response": response}

    except Exception as e:
        raise e
        return {"success": False, "error": str(e)}


@app.post("/vehicle-insights-v2")
async def vehicle_insights(request: Request):

    request = await request.body()
    request = json.loads(request)
    messages = request["messages"]
    model_name = request["model_name"]
    query = request["query"]
    search_index_name = AZURE_SEARCH_INDEX_NAME

    try:

        task_trigger = LLMRsponseGeneration(
            search_client, openai_client, search_index_client, search_index_name
        )

        # intents = await QueryAnalyzer.intents_extraction(task_trigger, query)

        search_term = await QueryAnalyzer.query_rephraser(messages, task_trigger, query)
        print("\n\nSEARCH TERM: ", search_term, "\n\n")
        cr = ContextRetrieval(search_client, openai_client)

        start_time = time.time()
        formatted_context, context = await cr.retrieve(search_term, model_name, 5)
        print(context)
        end_time = time.time()
        context_retrieval_time = end_time - start_time

        chat_history = []
        for i in range(len(messages)):
            chat_history.append({"role": "user", "content": messages[i]["query"]})
            chat_history.append(
                {"role": "assistant", "content": messages[i]["response"]}
            )

        start_time = time.time()
        prompt = PromptSet.doc_specific_prompt
        prompt = prompt.replace("{context}", context)
        context = (
            [{"role": "system", "content": prompt}]
            + chat_history
            + [{"role": "user", "content": query}]
        )
        response = await task_trigger.generate_responses(query, context)
        end_time = time.time()
        generated_response_time = end_time - start_time

        return {
            "success": True,
            "context": formatted_context,
            "context_retrieval_time": context_retrieval_time,
            "generated_response": response,
            "generated_response_time": generated_response_time,
            "rephrased_query": search_term,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# Allowing public access to the endpoints
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=4, reload=False)
