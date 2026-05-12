import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "resume-ner-bucket")
    
    # Security
    API_KEY: str = os.getenv("API_KEY", "default-dev-key")
    MAX_UPLOAD_SIZE: int = 5 * 1024 * 1024 # 5 MB

settings = Settings()
