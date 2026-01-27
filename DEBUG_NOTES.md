# Uwagi debugowe — "Downloading artifacts" i MVP

## Problem: "Downloading artifacts 0%" w terminalu Uvicorn

**Przyczyna:** Endpoint `/health` wywoływał `_get_or_init_rag()`, co przy pierwszym żądaniu budowało RAG i pobierał **wiele** artefaktów z MLflow (`load_mlflow_artifact_docs`). Każde `client.download_artifacts(...)` wyświetla w konsoli pasek postępu. Przy wolnej sieci lub MLflow na localhost:5050 powodowało to mnóstwo "Downloading artifacts" i opóźnienia.

**Wprowadzona zmiana:** W `src/api/app.py` endpoint `/health` **nie** uruchamia już budowy RAG. Używa tylko już załadowanego `app.state.rag`. RAG jest budowany dopiero przy pierwszym wywołaniu `/rag/ask` lub `/rag/answer`. Dzięki temu:
- pierwsze GET `/health` nie ściąga nic z MLflow,
- Streamlit nie triggeruje długich pobrań przy odświeżaniu Health.

## Szybszy start bez MLflow (cache)

Jeśli masz już skopiowane artefakty w `data/mlflow_cache/runs/<run_id>/`, API ładuje model z tego katalogu i **nie** odpytywa MLflow o model (tylko o expected_columns / drift_reference, jeśli ich brak w cache). Aby uniknąć jakichkolwiek połączeń do MLflow przy starcie, możesz:
- postawić MLflow z backendem plikowym (`mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts`) i mieć artefakty lokalnie,
- albo użyć **MVP** w folderze `mvp/` — bez MLflow, model z `joblib`.

## MVP (prosty projekt do rozmowy)

W folderze **`mvp/`** jest minimalna wersja:
- `python train.py` — uczenie, zapis do `mvp/artifacts/`
- `uvicorn api:app --host 0.0.0.0 --port 8000` — API (model z dysku, start od razu)
- `streamlit run streamlit_app.py` — UI

Brak MLflow, RAG i driftu przy starcie. Odpowiednie do pokazania na rozmowie (VELO Bank, Santander) jako „prosty, działający pipeline”.
