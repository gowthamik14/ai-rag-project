# Claude Code Instructions

## Commit Messages

- Do **not** include `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` in any commit messages.
- Keep commit messages concise and descriptive.

## Architecture

### SOLID principles — enforced at all times

| Principle | Rule |
|---|---|
| **Single Responsibility** | Routes handle HTTP only. Services handle one external concern each. |
| **Open/Closed** | New LLM backends are added by creating a new `LLMService` subclass, not by editing routes. |
| **Liskov Substitution** | Any `LLMService` subclass must be usable wherever `LLMService` is expected. |
| **Interface Segregation** | Service interfaces are small and focused (one method per concern). |
| **Dependency Inversion** | Routes depend on the `LLMService` abstraction via FastAPI `Depends`, never on `ChatOllama` directly. |

### Folder layout

```
api/         ← HTTP layer only (routes, request/response models)
services/    ← one file per external concern (LLM, vector store, etc.)
config/      ← settings read from environment
rag/         ← contains data loader and grpah.py (RAG pipeline Documentloader)
tests/       ← mirrors the source layout
```

### Adding a new service

1. Define an abstract base class in `services/<name>_service.py`
2. Implement the concrete class in the same file
3. Expose a `get_<name>_service()` function decorated with `@lru_cache(maxsize=1)`
4. Inject it into routes via `Depends(get_<name>_service)`

## Testing

Every endpoint and every component must have tests. No exceptions.

### Run tests
```bash
pytest
```

### Rules

- **Every new endpoint** → `tests/test_<feature>.py`
- **Every new service** → `tests/test_<name>_service.py`
- **Cover every line**: happy path, validation errors, and failure/error cases
- **Never call real external services** in tests — always use fakes or mocks

### Test doubles

Prefer a hand-written `Fake*` class over `MagicMock` when testing through the HTTP layer — it implements the real interface and records what it received:

```python
class FakeLLMService(LLMService):
    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.received_message: str | None = None
        self.call_count = 0

    def chat(self, message: str) -> str:
        self.received_message = message
        self.call_count += 1
        return self.reply
```

Use `MagicMock` + `patch` only when testing a service's internals directly (e.g. verifying `ChatOllama` was called correctly).

### Wiring fakes into FastAPI

Use `dependency_overrides` — never `patch` a module-level variable:

```python
@pytest.fixture()
def client(fake_service: FakeLLMService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_llm_service] = lambda: fake_service
    return TestClient(app, raise_server_exceptions=False)
```

### What to test for each endpoint

| Case | What to assert |
|---|---|
| Happy path | status 200, correct response shape, correct values |
| Forwarding | the right data was passed to the service |
| Call count | service was called exactly once |
| Validation | missing/wrong fields → 422 |
| Edge cases | empty strings, very long inputs |
| Errors | service raises → 500 |

### What to test for each service

| Case | What to assert |
|---|---|
| Output | returns the value from the underlying client |
| Forwarding | wraps input correctly before calling the client |
| Call count | client called exactly once |
| Configuration | reads model/URL from settings, not hardcoded |
| Singleton | `get_*_service()` returns the same instance every call |
