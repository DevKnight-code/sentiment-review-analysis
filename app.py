# Re-export the Flask app from the backend package.
# Gunicorn on Render uses: gunicorn app:app
from backend.app import app  # noqa: F401
