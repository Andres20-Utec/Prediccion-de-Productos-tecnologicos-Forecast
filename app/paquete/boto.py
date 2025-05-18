import os
import pandas as pd
from dotenv import load_dotenv
import boto3
from botocore.exceptions import NoCredentialsError
load_dotenv()
from dotenv import load_dotenv

class BotoClient:
    _instance = None

    def __init__(self):
        self.access_key = os.getenv('AWS_ACCESS_KEY')
        self.secret_key = os.getenv('AWS_SECRET_KEY')
        self.bucket_name = os.getenv('BUCKET_NAME')

        if not all([self.access_key, self.secret_key, self.bucket_name]):
            raise EnvironmentError("Faltan una o más variables de entorno requeridas: AWS_ACCESS_KEY, AWS_SECRET_KEY, BUCKET_NAME")
        
        self.session = boto3.Session(
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )

        self.client = self.session.client('s3')

    def upload_file_to_s3(self, file_path, key):
        try:
            self.client.upload_file(file_path, self.bucket_name, key)
            print(f"Archivo subido a S3: {key}")
        except NoCredentialsError as e:
            print(f"Error: Credenciales no encontradas - {e}")
        except Exception as e:
            print(f"Error inesperado: {e}")

    def download_file_from_s3(self, object_name, file_path):
        try:
            self.client.download_file(self.bucket_name, object_name, file_path)
            print(f"Archivo descargado de S3: {file_path}")
        except NoCredentialsError as e:
            print(f"Error: Credenciales no encontradas - {e}")
        except Exception as e:
            print(f"Error inesperado: {e}")

boto_client = BotoClient()
