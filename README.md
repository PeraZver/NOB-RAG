# AI Processing Source

This folder holds the Python scripts used to process books and generate machine-readable outputs.

## Scripts

`extract_pdf_to_jsonl.py`

Reads a PDF with `pdfplumber` one page at a time and writes a `pages.jsonl` file without loading the whole document into memory. Each JSONL record contains:

- `page_number`
- `text`
- `word_count`

The script also writes a `metadata.json` file for the processed book.

`chunk_pages_jsonl.py`

Reads `pages.jsonl` line by line and groups page text into overlapping chunks. By default it writes `chunks.jsonl` with chunks of about 800 words and 120 words of overlap. Each chunk record contains:

- `chunk_id`
- `source_pages`
- `text`
- `word_count`

`clean_pages_jsonl.py`

Cleans `pages.jsonl` before chunking. It dehyphenates print line-breaks, merges wrapped lines into cleaner paragraphs, removes obvious page-number noise, and moves likely footnote blocks out of the main text into a separate `footnotes` field.

`validate_chunks_jsonl.py`

Reads `chunks.jsonl` line by line and prints a summary of chunk sizes. It reports the total number of chunks, the minimum, maximum, and average chunk size in words, flags any chunks below a configurable threshold, and checks for:

- non-sequential `chunk_id` values
- empty `text`
- `word_count` mismatches
- broken overlap continuity between adjacent chunks

`rag_utils.py`

Shared helpers for the local RAG workflow: JSONL iteration, page parsing, Chroma collection naming, and page citation formatting.

`rag_env.py`

Loads local `.env` files and resolves provider-specific default models.

`rag_retrieval.py`

Holds the retrieval pipeline: Chroma access, embedding-model loading, lexical matching, metadata-aware boosts, reranking, and the reusable `RetrievalSession`.

`rag_providers.py`

Wraps the provider-specific answer-generation calls for OpenAI and Anthropic.

`rag_engine.py`

Provides the reusable `BookRAGEngine` that ties retrieval and answer generation together for one-shot and long-running scripts.

`rag_profiles.py`

Defines reusable query presets such as `--fast` and `--deep` so multiple CLI entrypoints can share the same retrieval behavior.

`build_chroma_index.py`

Builds a local persistent ChromaDB index from `chunks.jsonl` using `BAAI/bge-m3`. It stores chunk text plus metadata such as chunk id and source pages.

`ask_book_rag.py`

One-shot CLI entrypoint for the local RAG system. It uses the shared engine and retrieval modules, then sends only the strongest chunks to either the OpenAI API or the Anthropic API for a grounded answer with page citations.

`ask_book_rag_live.py`

Interactive CLI entrypoint that keeps Chroma, the embedder, and the optional reranker loaded in one long-running process. Ask a new question after the previous answer, and stop it with `Ctrl+C`.

`campaign_utils.py`

Helpers for staged campaign extraction: chunk batching, JSON parsing, event normalization, deduplication, and final document shaping.

`brigade_campaign_template.json`

Shared repo-level template for brigade campaign JSON outputs. All brigade extraction and verification runs can use this same baseline shape.

`build_brigade_campaign_json.py`

Extracts brigade campaign events from `chunks.jsonl` in resumable LLM-sized batches, saves intermediate batch JSON files, and consolidates them into a final brigade-campaign JSON document using the reference template shape.

`verify_brigade_campaign_json.py`

Runs a second-stage verification pass over extracted campaign events only. It groups likely duplicates, asks the provider to tighten dates, merge overlapping events more intelligently, and improve approximate coordinates, then consolidates the verified groups into a polished final JSON.

`requirements-rag.txt`

Minimal package list for the local RAG workflow.

## Folder Organization

`AI Processing/source/`

Stores the processing scripts and lightweight documentation.

Also stores the shared `brigade_campaign_template.json` used by campaign extraction and verification.

`AI Processing/<book title>/`

Stores the output for one book. Right now that includes:

- `pages.jsonl`: one JSON object per page
- `chunks.jsonl`: one JSON object per chunk
- `metadata.json`: basic information about the source PDF and output location
- `chroma_db/`: local persistent vector store for retrieval

## Examples

Run the extractor from the project root:

```powershell
python ".\AI Processing\source\extract_pdf_to_jsonl.py" ".\SFRJ literatura\Milarn Rako_Družjanič -11. dalmatinska brigada.pdf"
```

Then build overlapping chunks:

```powershell
python ".\AI Processing\source\clean_pages_jsonl.py" ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada\pages.jsonl"
python ".\AI Processing\source\chunk_pages_jsonl.py" ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada\cleaned_pages.jsonl"
```

Validate the chunk output:

```powershell
python ".\AI Processing\source\validate_chunks_jsonl.py" ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada\chunks.jsonl"
```

Build the local Chroma index:

```powershell
python ".\AI Processing\source\build_chroma_index.py" ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada" --reset
```

Ask a question against the indexed book:

```powershell
python ".\AI Processing\source\ask_book_rag.py" ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada" "Kada i gdje je osnovana 11. dalmatinska brigada?" --show-sources
```

Use OpenAI instead:

```powershell
python ".\AI Processing\source\ask_book_rag.py" ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada" "Kada i gdje je osnovana 11. dalmatinska brigada?" --provider openai --show-sources
```

Preview retrieval without calling OpenAI:

```powershell
python ".\AI Processing\source\ask_book_rag.py" ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada" "Kada i gdje je osnovana 11. dalmatinska brigada?" --retrieve-only
```

Run the long-lived interactive CLI:

```powershell
python ".\AI Processing\source\ask_book_rag_live.py" ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada" --show-sources
```

Use the faster preset for everyday interactive querying:

```powershell
python ".\AI Processing\source\ask_book_rag_live.py" ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada" --fast
```

Use the deeper preset when you want broader retrieval and reranking:

```powershell
python ".\AI Processing\source\ask_book_rag_live.py" ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada" --deep --show-sources
```

Run the interactive CLI in retrieval-only mode:

```powershell
python ".\AI Processing\source\ask_book_rag_live.py" ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada" --retrieve-only
```

Extract campaign data in resumable batches:

```powershell
python ".\AI Processing\source\build_brigade_campaign_json.py" extract ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada" --batch-size 8
```

Continue only a slice of batches:

```powershell
python ".\AI Processing\source\build_brigade_campaign_json.py" extract ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada" --batch-size 8 --start-batch 6 --end-batch 10
```

Consolidate finished batch files into the final JSON:

```powershell
python ".\AI Processing\source\build_brigade_campaign_json.py" consolidate ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada"
```

Run the second-stage verifier on extracted events:

```powershell
python ".\AI Processing\source\verify_brigade_campaign_json.py" verify ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada"
```

Then consolidate the verified groups:

```powershell
python ".\AI Processing\source\verify_brigade_campaign_json.py" consolidate ".\AI Processing\Rako Druzjanic -11. dalmatinska brigada"
```

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
- The Anthropic fallback default is `claude-sonnet-4-6` unless `ANTHROPIC_MODEL` is set in `.env`.
- The query script sends only the top reranked chunks to the model, not the whole book.
- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` must be set in the environment for answer generation. The query script also auto-loads `.env` from the current working directory, `AI Processing/source/`, or the book directory.
- The first run of the embedding and reranker models may download model files from Hugging Face.
