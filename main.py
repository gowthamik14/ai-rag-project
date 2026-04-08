"""
Entry point for the AI RAG Service.

Run modes
---------
  API server (default):
    python main.py

  One-shot CLI query:
    python main.py query "What is the capital of France?"

  Ingest a file:
    python main.py ingest path/to/document.txt

  Summarise a stored document:
    python main.py summarize --doc-id <doc_id> --style bullet_points
"""
from __future__ import annotations

import sys


def run_server() -> None:
    import uvicorn
    from config.settings import settings
    from api.app import create_app

    app = create_app()
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower(),
    )


def run_query(question: str) -> None:
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    result = pipeline.query(question)
    print("\n" + "=" * 60)
    print(f"Q: {result.question}")
    print(f"\nA: {result.answer}")
    print(f"\nSources ({len(result.retrieved_chunks)}):")
    for i, chunk in enumerate(result.retrieved_chunks, 1):
        print(f"  [{i}] score={chunk.get('score', 0):.3f}  source={chunk.get('source', '')}")
    print("=" * 60)


def run_ingest(file_path: str) -> None:
    from workflows.ingestion_workflow import IngestionWorkflow

    result = IngestionWorkflow().run({"file_path": file_path})
    if result.status.value == "failed":
        print(f"Ingestion failed: {result.errors}")
        sys.exit(1)
    print(f"Ingested: doc_id={result.output['doc_id']}  chunks={result.output['chunks_indexed']}")


def run_summarize(doc_id: str | None, text: str | None, style: str) -> None:
    from workflows.summarization_workflow import SummarizationWorkflow

    inputs: dict = {"style": style}
    if doc_id:
        inputs["doc_id"] = doc_id
    elif text:
        inputs["text"] = text
    else:
        print("Provide --doc-id or --text")
        sys.exit(1)

    result = SummarizationWorkflow().run(inputs)
    if result.status.value == "failed":
        print(f"Summarization failed: {result.errors}")
        sys.exit(1)
    print("\n" + "=" * 60)
    print(result.output["summary"])
    print("=" * 60)


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] == "serve":
        run_server()
        return

    cmd = args[0]

    if cmd == "query":
        question = " ".join(args[1:])
        if not question:
            print("Usage: python main.py query <question>")
            sys.exit(1)
        run_query(question)

    elif cmd == "ingest":
        if len(args) < 2:
            print("Usage: python main.py ingest <file_path>")
            sys.exit(1)
        run_ingest(args[1])

    elif cmd == "summarize":
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--doc-id", default=None)
        parser.add_argument("--text", default=None)
        parser.add_argument("--style", default="concise", choices=["concise", "detailed", "bullet_points"])
        parsed = parser.parse_args(args[1:])
        run_summarize(parsed.doc_id, parsed.text, parsed.style)

    else:
        print(f"Unknown command: {cmd}")
        print("Commands: serve | query | ingest | summarize")
        sys.exit(1)


if __name__ == "__main__":
    main()
