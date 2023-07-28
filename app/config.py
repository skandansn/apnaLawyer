from pydantic import BaseSettings


class Settings(BaseSettings):
    DATABASE_PORT: int = 6500
    POSTGRES_PASSWORD: str = "pass"
    POSTGRES_USER: str = "user"
    POSTGRES_DB: str = "apnalawyer"
    POSTGRES_HOST: str = "127.0.0.1"
    POSTGRES_HOSTNAME: str = "localhosts"

    JWT_PUBLIC_KEY: str = ""
    JWT_PRIVATE_KEY: str = ""
    REFRESH_TOKEN_EXPIRES_IN: int = 60
    ACCESS_TOKEN_EXPIRES_IN: int = 3600
    JWT_ALGORITHM: str = "RS256"

    CLIENT_ORIGIN: str = "http://localhost:3000"

    OPENAI_API_KEY: str = "openaikey"

    PINECONE_ENV: str = "pineconeenv"
    PINECONE_API_KEY: str = "pineconekey"
    PINECONE_INDEX: str = "pineconeindex"

    KANOON_API_TOKEN: str = "kanoonkey"
    KANOON_API_URL = 'kanoon_url'

    class Config:
        env_file = './.env'


settings = Settings()
