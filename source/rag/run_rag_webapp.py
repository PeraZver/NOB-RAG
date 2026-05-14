from __future__ import annotations

import os

import uvicorn
from rag.rag_webapp import app


def main() -> None:
    host = os.getenv("RAG_WEBAPP_HOST", "127.0.0.1")
    port = int(os.getenv("RAG_WEBAPP_PORT", "8000"))
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
