"""Load and parse configuration from YAML."""
from pathlib import Path
import yaml
from pydantic import BaseModel


class DatabaseConfig(BaseModel):
    """Database configuration."""
    url: str
    echo: bool = False


class CrawlerConfig(BaseModel):
    """Crawler configuration."""
    timeout: int = 30
    batch_size: int = 1000


class Settings(BaseModel):
    """Application settings loaded from config.yaml."""
    database: DatabaseConfig
    crawler: CrawlerConfig
    
    @classmethod
    def from_yaml(cls, path: str | Path = None) -> "Settings":
        """Load settings from YAML file."""
        if path is None:
            path = Path(__file__).parent.parent.parent.parent / "config.yaml"
        
        with open(path) as f:
            data = yaml.safe_load(f)
        
        return cls(**data)
    
    @property
    def database_url(self) -> str:
        """Database URL for SQLAlchemy."""
        return self.database.url
    
    @property
    def db_echo(self) -> bool:
        """Whether to echo SQL statements."""
        return self.database.echo


# Load settings on module import
_settings = None

def get_settings() -> Settings:
    """Get or load settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings.from_yaml()
    return _settings
