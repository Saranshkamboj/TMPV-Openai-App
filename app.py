from dotenv import load_dotenv

load_dotenv(override=True)

import os
import time
import re
from datetime import datetime
import json
import uvicorn
from fastapi import FastAPI, Form, Request, Response
import logging
from fastapi import FastAPI, UploadFile, Form, BackgroundTasks
from typing import List
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

"""Following are the different logs at different routes which saves information and errors at different locations and saved in the logs directory."""
loggers = {}
for route in [
    "Vehicles-Logs",
]:
    logger = logging.getLogger(route)
    logger.setLevel(logging.ERROR)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler = logging.FileHandler(f"./logs/{route}.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    loggers[route] = logger
    logger = ""


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


async def upload_file(file, contents, model_name):
    if not file:
        return {"success": False, "error": "No file was uploaded."}

    try:
        pipeline = Pipeline()

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
            print("File Indexed")

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/uploadfile")
async def submit_form(
    background_tasks: BackgroundTasks, file: UploadFile, model_name: str = Form()
):
    try:
        file_name = file.filename
        contents = await file.read()
        file_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_{file_name}"
        background_tasks.add_task(upload_file, file, contents, model_name)
    except Exception as e:
        loggers["Vehicles-Logs"].exception(f"Exception occurred. {e}")
    return f"Uploading {file_name}"


@app.post("/delete-records")
async def deelte_records(id: str = Form(), no_of_records: int = Form()):
    for i in range(no_of_records):
        doc_id = f"{id}_{i+1}_{i+1}"
        documents = {"id": doc_id}
        delete_documents_batch = search_client.delete_documents(documents=documents)
        print("deleted", id)


@app.post("/vehicle-insights-v2")
async def vehicle_insights(request: Request):

    request = await request.body()
    request = json.loads(request)
    messages = request["messages"]
    model_name = request["model_name"]
    query = request["query"]
    search_index_name = AZURE_SEARCH_INDEX_NAME

    try:
        completion_tokens = 0
        prompt_tokens = 0

        task_trigger = LLMRsponseGeneration(
            search_client, openai_client, search_index_client, search_index_name
        )

        response = await QueryAnalyzer.intents_extraction(task_trigger, query)
        intents = json.loads(response["response"])["entities"]
        completion_tokens += response["completion_tokens"]
        prompt_tokens += response["prompt_tokens"]

        query = ""
        if "others" in intents.keys() and len(intents.keys()) == 1:
            response = "I'm sorry, but I am a Tata Motors chatbot and I am here to assist you with queries related to Tata Motors cars."
            return {
                "success": True,
                "context": [],
                "context_retrieval_time": 0,
                "generated_response": response,
                "generated_response_time": 0,
                "rephrased_query": "",
            }

        for k, v in intents.items():
            if k == "others":
                continue
            query += " " + v

        search_term, completion_tokens, prompt_tokens = (
            await QueryAnalyzer.query_rephraser(messages, task_trigger, query)
        )
        completion_tokens += completion_tokens
        prompt_tokens += prompt_tokens
        cr = ContextRetrieval(search_client, openai_client)

        start_time = time.time()
        formatted_context, context = await cr.retrieve(search_term, model_name, 5)
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
            chat_history
            + [{"role": "system", "content": prompt}]
            + [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        )
        response = await task_trigger.generate_responses(query, context)
        end_time = time.time()
        generated_response_time = end_time - start_time
        completion_tokens += response["completion_tokens"]
        prompt_tokens += response["prompt_tokens"]

        return {
            "success": True,
            "context": formatted_context,
            "context_retrieval_time": context_retrieval_time,
            "generated_response": response["response"],
            "generated_response_time": generated_response_time,
            "rephrased_query": search_term,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    except Exception as e:
        loggers["Vehicles-Logs"].exception(f"Exception occurred. {e}")
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
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=4, reload=True)
