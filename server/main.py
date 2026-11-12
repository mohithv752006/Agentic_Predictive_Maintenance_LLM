"""Start the API server: ``python main.py`` (same as ``python -m scripts.serve``).

Options: ``python main.py --host 127.0.0.1 --port 8000 --config extra.yaml``.
Run from the ``server/`` folder with the virtual environment activated.
"""

from scripts.serve import app

if __name__ == "__main__":
    app()
