# gentext-app

A FastAPI app that generates text with a bigram model and serves spaCy word embeddings (`en_core_web_md`, 300 dimensions).

## Run with Docker

```bash
docker build -t gentext-app .
docker run -p 8000:8000 gentext-app
```

Then open http://127.0.0.1:8000/docs to try the endpoints, or query them directly:

```bash
curl -X POST http://127.0.0.1:8000/generate -H "Content-Type: application/json" -d '{"start_word": "the", "length": 6}'
curl -X POST http://127.0.0.1:8000/embedding -H "Content-Type: application/json" -d '{"word": "apple"}'
```

## Run locally (without Docker)

```bash
uv sync
uv run fastapi dev app/main.py
```

## Endpoints

| Method | Path          | Body                                   | Returns                                  |
|--------|---------------|----------------------------------------|------------------------------------------|
| GET    | `/`           | none                                   | `{"Hello": "World"}`                     |
| POST   | `/generate`   | `{"start_word": "the", "length": 6}`   | Text generated from bigram probabilities |
| POST   | `/embedding`  | `{"word": "apple"}`                    | The word's 300-dimensional vector        |
