from azure.storage.blob import ContainerClient
import os

def upload(files, connection_string, container_name):
    container_client = ContainerClient.from_connection_string(connection_string, container_name)
    print('Uploading files to blob storage...')

    for file in files:
        blob_client = container_client.get_blob_client(file.name)

        with open(file.path, 'rb') as data:
            blob_client.upload_blob(data, overwrite=True)
            print(f"{file.name} uploaded to blob storage")


def get_files(dir):
    with os.scandir(dir) as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith('.'):
                yield entry


