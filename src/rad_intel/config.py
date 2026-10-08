"""
Configuration management for Rad-Intel using Pydantic Settings.
"""

from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project metadata
    BASE_DIR: Path = BASE_DIR
    PROJECT_NAME: str = "Rad-Intel"
    VERSION: str = "0.1.0"
    DEBUG: bool = False

    # Gemini API configuration
    GEMINI_API_KEY: str | None = Field(default=None)
    GEMINI_MODEL: str = "gemini-3.5-flash-lite"

    # Hardware & Model execution
    DEVICE: Literal["auto", "cpu", "cuda"] = "auto"
    DEFAULT_MODEL: Literal[
        "hybrid", "densenet121", "swin_t", "hybrid_no_cbam", "resnet50"
    ] = "hybrid"
    MODEL_WEIGHTS_PATH: str | None = None
    NUM_CLASSES: int = 2
    CLASS_NAMES: list[str] = Field(default_factory=lambda: ["NORMAL", "PNEUMONIA"])

    # Image Preprocessing
    IMAGE_SIZE: int = 224
    NORM_MEAN: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    NORM_STD: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])
    APPLY_CLAHE: bool = True

    # Server parameters
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    # Storage paths
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR / 'rad_intel.db'}"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"

    @property
    def torch_device(self) -> torch.device:
        """Resolve actual PyTorch compute device."""
        if self.DEVICE == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self.DEVICE == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")


# Global cached settings instance
settings = Settings()
