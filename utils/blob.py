
"""Blob storage operation functionalities implementation"""
from azure.storage.blob import BlobServiceClient
import json
import os
from ast import literal_eval

"""Setting up blob storage container"""
connect_str = os.getenv('BLOB_STORAGE_ENDPOINT')

container_name = "vehicle-chat-history"
blob_name_prefix = "context_"
blob_service_client = BlobServiceClient.from_connection_string(
    connect_str)
container_client = blob_service_client.get_container_client(container_name)


async def connect_blob(chat_session_id: str, query: str, data=None):
    
    """Setting up blob storage client"""
    blob_name = blob_name_prefix + chat_session_id
    blob_client = container_client.get_blob_client(blob_name)

    if query == 'blob_status':
        """Check if blob is exist or not"""
        blob_exists = blob_client.exists()
        return blob_exists
    
    elif query == 'download':
        """Blob already exists, read its contents and perform operations"""
        blob_contents = blob_client.download_blob().content_as_text()
        chat_session_data = json.loads(blob_contents)
        return chat_session_data
    
    elif query == 'upload':
        """Upload on Blob Storage with sliding window optimization"""
        if len(data)>10:
            del data[1]
            del data[1]

        blob_client.upload_blob(json.dumps(data), overwrite=True)
