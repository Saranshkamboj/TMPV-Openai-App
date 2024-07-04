import os
import aiofiles
import json
from azure.storage.blob import (
    BlobServiceClient,
    BlobClient,
    ContainerClient,
    generate_blob_sas,
    BlobSasPermissions,
)
from datetime import datetime, timedelta


class BlobStorage:
    def __init__(self, blob_storage_client):
        print("INITIALIZING BlobStorage() ....")
        self.blob_service_client = blob_storage_client
        self.container_name = "vehicles-data"

        self.blob_service_name = ""

    async def upload_to_blob_storage(self, file_path):
        try:

            try:
                container_client = self.blob_service_client.get_container_client(
                    self.container_name
                )
                await container_client.create_container()
            except Exception as e:
                if "ContainerAlreadyExists" not in str(e):
                    raise Exception(f"Failed to create container: {str(e)}")

            file_name = os.path.basename(file_path)

            try:
                blob_client = container_client.get_blob_client(file_name)
                async with aiofiles.open(file_path, "rb") as data:
                    await blob_client.upload_blob(await data.read(), overwrite=True)
            except Exception as e:
                raise Exception(f"Failed to upload blob: {str(e)}")
            sas_url = self.generate_sas_url(file_name)
            file_url = blob_client.url
            print(sas_url)
            # await self.blob_service_client.close()
            return sas_url

        except Exception as e:
            print(f"Error in upload_to_blob_storage: {str(e)}")
            raise e

    async def upload_json(self, data, filename):
        try:
            container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
            blob_client = container_client.get_blob_client(filename)
            # Check if the blob already exists
            # print(data)
            if await blob_client.exists():
                # Download existing data
                try:
                    existing_data = await blob_client.download_blob()
                    existing_data = await existing_data.readall()
                except:

                    existing_data = blob_client.download_blob().readall()

                existing_json = json.loads(existing_data)

            else:
                existing_json = []

            # Append new data to existing JSON
            existing_json.append(data)

            # Convert to JSON string
            updated_json_data = json.dumps(existing_json)

            # Upload updated JSON to blob
            await blob_client.upload_blob(updated_json_data, overwrite=True)
            print("Blob updated successfully.")

        except Exception as e:
            raise e
            print(f"Error: {str(e)}")

    def generate_sas_url(self, blob_name):

        container_client = self.blob_service_client.get_container_client(
            self.container_name
        )
        blob_client = container_client.get_blob_client(blob_name)
        try:
            sas_token = generate_blob_sas(
                account_name=self.blob_service_client.account_name,
                container_name=self.container_name,
                blob_name=blob_name,
                account_key=self.blob_service_client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow()
                + timedelta(hours=1),  # SAS URL valid for 1 hour
            )
            sas_url = f"https://{self.blob_service_client.account_name}.blob.core.windows.net/{self.container_name}/{blob_name}?{sas_token}"
            print(sas_url)
            # exit(0)
            return sas_url

        except Exception as e:
            print(f"Error generating SAS URL: {str(e)}")
            return None


# Example usage (to be run in an async context):
# async def main():
#     account_name = os.environ.get("BLOB_STORAGE_ACCOUNT_NAME")
#     account_key = os.environ.get("BLOB_STORAGE_ACCOUNT_KEY")
#     file_path = "data_ingestion_pipleine\ATech - Prospectus for exposure.pdf"
#     metadata = {"key": "value"}
#     blob_storage = BlobStorage(account_name, account_key)
#     url = await blob_storage.upload_to_blob_storage(file_path, metadata, TYPE="IE")
#     print(f"File URL: {url}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
