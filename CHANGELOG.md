# Changelog - Fixes for Interview Demo

## Naprawione problemy (26 stycznia 2026)

### 🔧 Główne naprawy

1. **Model Loader - Lepsze cache i timeout handling**
   - ✅ Model jest cache'owany lokalnie po pierwszym pobraniu
   - ✅ Timeout handling (30-120s) żeby uniknąć zawieszenia
   - ✅ Graceful fallback jeśli MLflow jest niedostępny
   - ✅ Lepsze error handling z informatywnymi komunikatami

2. **RAG Service - Lazy loading**
   - ✅ RAG nie blokuje startu API
   - ✅ Inicjalizacja na pierwszym żądaniu (lazy loading)
   - ✅ Timeout protection przy pobieraniu artefaktów MLflow
   - ✅ Graceful degradation - działa nawet bez MLflow artifacts

3. **API - Lepsze error handling**
   - ✅ 503 tylko gdy model rzeczywiście się ładuje
   - ✅ Lepsze komunikaty błędów
   - ✅ Health check nie blokuje podczas ładowania
   - ✅ RAG endpoints z lepszym error handling

4. **Streamlit - Zwiększone timeouty**
   - ✅ Timeout zwiększony z 10s do 30-60s
   - ✅ Lepsze komunikaty timeout
   - ✅ Graceful handling 503 errors

### 📦 Nowe pliki

1. **setup.sh / setup.ps1** - Automatyczny setup projektu
   - Tworzy venv
   - Instaluje zależności
   - Pobiera dataset
   - Tworzy potrzebne katalogi

2. **INTERVIEW_PREP.md** - Kompletne przygotowanie do rozmów
   - Demo script
   - Top pytania rekrutacyjne
   - Odpowiedzi na kluczowe pytania
   - Plan działania

3. **CHANGELOG.md** - Ten plik

### 📝 Zaktualizowane pliki

1. **README.md** - Lepsze instrukcje
   - Quick start dla demo
   - Troubleshooting section
   - Offline mode instructions
   - Pre-warming cache instructions

2. **requirements.txt** - Dodano joblib

### 🎯 Co teraz zrobić?

1. **Przetestuj projekt:**
   ```bash
   # Windows
   .\setup.ps1
   
   # Linux/Mac
   ./setup.sh
   ```

2. **Przygotuj demo:**
   - Przejdź przez README Quick Start
   - Przećwicz prezentację z INTERVIEW_PREP.md
   - Przetestuj pełny flow przed rozmową

3. **Przed rozmową:**
   - Pre-warm cache (zrób kilka predictions)
   - Sprawdź że wszystko działa
   - Przygotuj demo script

### ⚠️ Znane ograniczenia

- Timeout handling jest "best-effort" - MLflow client może nie zawsze respektować timeout
- RAG artifacts z MLflow są opcjonalne - jeśli nie działają, RAG używa tylko markdown docs
- Windows timeout: używamy threading zamiast signal (signal.SIGALRM nie działa na Windows)

### 🚀 Następne kroki (opcjonalne)

- [ ] Dodaj unit testy
- [ ] Dodaj integration testy
- [ ] Docker compose dla całego stacku
- [ ] Prometheus metrics
- [ ] Lepsze logging

---

**Projekt jest gotowy do demo! 🎉**
