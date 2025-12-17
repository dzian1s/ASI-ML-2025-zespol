from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_PATH: str | None = None
    DATABASE_URL: str
    MODEL_VERSION: str = "unknown"

    class Config:
        env_file = ".env"


settings = Settings()
