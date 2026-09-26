"""Verify the API key works by listing models (GET /v1/models, not billed). Never prints the key."""

from typesafe_sdk import TypeSafeClient, TypeSafeError

from jev_benchmarking.config import API_KEY_FILE, MODEL, load_api_key


def main() -> None:
    try:
        with TypeSafeClient(api_key=load_api_key()) as client:
            models = client.models.list().models
    except TypeSafeError as e:
        raise SystemExit(f"API check failed: {type(e).__name__}: {e}") from None
    print(f"Key OK (source: $TYPESAFE_API_KEY or {API_KEY_FILE}). Models visible to this account:")
    for m in models:
        print(f"  {m.name:<14} released {m.release_date}  {m.description}")
    print(f"Benchmarks are pinned to: {MODEL}")


if __name__ == "__main__":
    main()
