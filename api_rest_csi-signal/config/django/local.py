# Load the default config
from .base import *

# Set up .env
from config.env import env

# Set up to work with everything
DEBUG = True
ALLOWED_HOSTS = ["*"]

# CORS_ALLOWED_ORIGINS = [*]
CORS_ALLOW_ALL_ORIGINS = True


# DB SETTINGS
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

