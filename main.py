"""CLI entry point for radiation safety RAG queries."""

from dotenv import load_dotenv

load_dotenv()


def main():
    """Interactive CLI for querying the RAG system."""
    from graph.graph import app

    print("Radiation Safety RAG - CLI")
    print("Type a question and press Enter. Empty to quit.\n")
    while True:
        try:
            q = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        result = app.invoke(
            {
                "question": q,
                "generation": "",
                "web_search": False,
                "documents": [],
                "web_search_attempted": False,
                "chat_history": [],
            }
        )
        print("\nAnswer:", result.get("generation", "(no answer)"))
        docs = result.get("documents", [])
        if docs:
            print(
                "Sources:",
                [getattr(d, "metadata", {}).get("source", "?") for d in docs],
            )
        print()


if __name__ == "__main__":
    main()
