# AI Processing Source

This folder holds the Python tooling used to process books, build local RAG indexes, run a browser UI, and extract structured campaign data.

## Code Layout

- `AI Processing/source/rag/`
  Retrieval engine, provider wrappers, Chroma indexing, interactive querying, and the browser app.
- `AI Processing/source/text_processing/`
  PDF extraction, page cleanup, chunking, and chunk validation.
- `AI Processing/source/campaigns/`
  Brigade campaign extraction, verification, polishing, and the shared JSON template.

## Main Components

`text_processing/extract_pdf_to_jsonl.py`

Reads a PDF with `pdfplumber` one page at a time and writes `pages.jsonl` plus `metadata.json`.

`text_processing/clean_pages_jsonl.py`

Cleans `pages.jsonl` before chunking by fixing print-layout artifacts and footnote noise.

`text_processing/chunk_pages_jsonl.py`

Builds overlapping retrieval chunks from cleaned pages and writes `chunks.jsonl`.

`text_processing/validate_chunks_jsonl.py`

Checks chunk sizes, continuity, and basic JSONL consistency.

`rag/build_chroma_index.py`

Builds the local ChromaDB index from `chunks.jsonl` using `BAAI/bge-m3`.

`rag/ask_book_rag.py`

One-shot CLI querying for a single indexed book.

`rag/ask_book_rag_live.py`

Long-lived interactive CLI that keeps retrieval models loaded between questions.

`rag/rag_webapp.py`

FastAPI app that serves the browser UI and API endpoints for book selection and chat.

`rag/run_rag_webapp.py`

Launcher for the browser UI with `RAG_WEBAPP_HOST` and `RAG_WEBAPP_PORT` support.

`campaigns/build_brigade_campaign_json.py`

Extracts brigade campaign events from chunked book text in resumable LLM-sized batches.

`campaigns/verify_brigade_campaign_json.py`

Runs a second verification pass that merges duplicates, tightens dates, and improves coordinates.

`campaigns/polish_verified_campaign_json.py`

Polishes already-verified events by shortening operation titles and tightening notes.

`requirements-rag.txt`

Package list for the local RAG workflow and browser UI.

## Book Output Folders

`AI Processing/<book title>/`

Stores the output for one book. Right now that includes:

- `pages.jsonl`: one JSON object per page
- `chunks.jsonl`: one JSON object per chunk
- `metadata.json`: basic information about the source PDF and output location
- `chroma_db/`: local persistent vector store for retrieval

## Usage

All commands below assume you first switch into the source root:

```powershell
cd ".\AI Processing\source"
```

Run the extractor from the project root:

```powershell
python -m text_processing.extract_pdf_to_jsonl "..\..\SFRJ literatura\Milarn Rako_DruÅ¾janiÄ -11. dalmatinska brigada.pdf"
```

Then build overlapping chunks:

```powershell
python -m text_processing.clean_pages_jsonl "..\Rako Druzjanic -11. dalmatinska brigada\pages.jsonl"
python -m text_processing.chunk_pages_jsonl "..\Rako Druzjanic -11. dalmatinska brigada\cleaned_pages.jsonl"
```

Validate the chunk output:

```powershell
python -m text_processing.validate_chunks_jsonl "..\Rako Druzjanic -11. dalmatinska brigada\chunks.jsonl"
```

Build the local Chroma index:

```powershell
python -m rag.build_chroma_index "..\Rako Druzjanic -11. dalmatinska brigada" --reset
```

Ask a question against the indexed book:

```powershell
python -m rag.ask_book_rag "..\Rako Druzjanic -11. dalmatinska brigada" "Kada i gdje je osnovana 11. dalmatinska brigada?" --show-sources
```

Use OpenAI instead:

```powershell
python -m rag.ask_book_rag "..\Rako Druzjanic -11. dalmatinska brigada" "Kada i gdje je osnovana 11. dalmatinska brigada?" --provider openai --show-sources
```

Preview retrieval without calling OpenAI:

```powershell
python -m rag.ask_book_rag "..\Rako Druzjanic -11. dalmatinska brigada" "Kada i gdje je osnovana 11. dalmatinska brigada?" --retrieve-only
```

Run the long-lived interactive CLI:

```powershell
python -m rag.ask_book_rag_live "..\Rako Druzjanic -11. dalmatinska brigada" --show-sources
```

Use the faster preset for everyday interactive querying:

```powershell
python -m rag.ask_book_rag_live "..\Rako Druzjanic -11. dalmatinska brigada" --fast
```

Use the deeper preset when you want broader retrieval and reranking:

```powershell
python -m rag.ask_book_rag_live "..\Rako Druzjanic -11. dalmatinska brigada" --deep --show-sources
```

Run the interactive CLI in retrieval-only mode:

```powershell
python -m rag.ask_book_rag_live "..\Rako Druzjanic -11. dalmatinska brigada" --retrieve-only
```

Extract campaign data in resumable batches:

```powershell
python -m campaigns.build_brigade_campaign_json extract "..\Rako Druzjanic -11. dalmatinska brigada" --batch-size 8
```

Continue only a slice of batches:

```powershell
python -m campaigns.build_brigade_campaign_json extract "..\Rako Druzjanic -11. dalmatinska brigada" --batch-size 8 --start-batch 6 --end-batch 10
```

Consolidate finished batch files into the final JSON:

```powershell
python -m campaigns.build_brigade_campaign_json consolidate "..\Rako Druzjanic -11. dalmatinska brigada"
```

Run the second-stage verifier on extracted events:

```powershell
python -m campaigns.verify_brigade_campaign_json verify "..\Rako Druzjanic -11. dalmatinska brigada"
```

Then consolidate the verified groups:

```powershell
python -m campaigns.verify_brigade_campaign_json consolidate "..\Rako Druzjanic -11. dalmatinska brigada"
```

Launch the browser-based interface:

```powershell
python -m rag.run_rag_webapp
```

Then open:

```text
http://127.0.0.1:8000
```

The web app will list every book folder under `AI Processing/` that contains both `chunks.jsonl` and `chroma_db/`.

## Ubuntu Migration

Copy the whole project folder to the Ubuntu machine so the processed book directories, `chunks.jsonl`, and `chroma_db/` come over together.

Recommended setup:

```bash
cd /path/to/NOB/AI\ Processing/source
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-rag.txt
```

Set your API key in a `.env` file at the project root, for example:

```dotenv
ANTHROPIC_API_KEY=...
RAG_PROVIDER=anthropic
ANTHROPIC_MODEL=claude-sonnet-4-6
```

or:

```dotenv
OPENAI_API_KEY=...
RAG_PROVIDER=openai
OPENAI_MODEL=gpt-5.3-codex
```

Start the browser UI on Ubuntu:

```bash
cd /path/to/NOB/AI\ Processing/source
python3 -m rag.run_rag_webapp
```

If you want it reachable from other devices on your network, bind to all interfaces:

```bash
cd /path/to/NOB/AI\ Processing/source
RAG_WEBAPP_HOST=0.0.0.0 RAG_WEBAPP_PORT=8000 python3 -m rag.run_rag_webapp
```

Then open `http://<ubuntu-ip>:8000` in the browser.

## RAG Notes

- Retrieval happens locally with ChromaDB and `BAAI/bge-m3`.
- The query script merges semantic retrieval with a lightweight lexical pass, then reranks the merged candidates with `BAAI/bge-reranker-v2-m3`.
- The retriever also applies metadata-aware boosts for chapter-heading chunks, contents-like chunks, and early historical sections when the question is about formation, chronology, commanders, or unit structure.
- Campaign-style questions about movements, locations, assaults, battles, and operations now boost operational narrative chunks and down-rank appendix-like personnel sections.
- The current query defaults use `800`-word chunks, `120`-word overlap, `40` retrieval candidates, and `8` final chunks sent to the model.
- `--fast` uses `top_k=4`, `candidate_k=12`, and skips reranking for quicker interactive use.
- `--deep` uses `top_k=10`, `candidate_k=60`, and keeps reranking on for more exhaustive retrieval.
- Cleaned pages generally work better for retrieval than raw `pdfplumber` page text because print-layout hyphenation and footnote noise are reduced before chunking.
- Generation happens through the selected provider SDK.
- Answer generation can use either OpenAI or Anthropic, selected with `--provider`. Anthropic is the default unless `RAG_PROVIDER` says otherwise.
- `ask_book_rag_live.py` is the faster choice for repeated questions because it keeps the models and Chroma client alive instead of reloading them for every query.
- `rag_webapp.py` also reuses cached engines inside the server process, so repeated questions against the same book avoid repeated startup costs.
- The Anthropic fallback default is `claude-sonnet-4-6` unless `ANTHROPIC_MODEL` is set in `.env`.
- The query script sends only the top reranked chunks to the model, not the whole book.
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` must be set in the environment for answer generation. The query script also auto-loads `.env` from the current working directory, `AI Processing/source/`, or the book directory.
- The first run of the embedding and reranker models may download model files from Hugging Face.
