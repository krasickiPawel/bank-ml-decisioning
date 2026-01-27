# Przygotowanie do rozmów rekrutacyjnych - AI/ML Expert

## 📅 Timeline
- **Dziś:** 26 stycznia 2026
- **Rozmowy:** 29 stycznia (czwartek) - Santander + VeloBank
- **Nokia:** 28 stycznia (środa)

---

## 🎯 Strategia przygotowania (Max ROI)

### PRIORYTET 1: Projekt Demo (Bank ML Decisioning) ✅
**Status:** Projekt naprawiony, gotowy do demo

**Co zrobić TERAZ:**
1. ✅ Projekt działa lokalnie offline
2. ⏳ **URGENT:** Przetestuj pełny flow przed rozmową:
   ```bash
   # Terminal 1: MLflow
   mlflow server --host 0.0.0.0 --port 5050 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlflow/artifacts
   
   # Terminal 2: Train models
   python -m src.train --experiment bank-credit-risk-http --run-name rf-baseline --model rf
   python src/scripts/select_best_model.py
   
   # Terminal 3: API
   uvicorn src.api.app:app --reload --port 8000
   
   # Terminal 4: Streamlit
   streamlit run streamlit_app.py
   ```
3. ⏳ **URGENT:** Przygotuj "demo script" - co pokazać w jakiej kolejności
4. ⏳ **URGENT:** Przećwicz opowiadanie o projekcie (5-10 min prezentacja)

### PRIORYTET 2: Przygotowanie merytoryczne (27-28 stycznia)
**Co się uczyć:**
- ✅ Overfitting - techniki, jak unikać, jak pokazać w projekcie
- ✅ Problem framing - jak podejść do problemu biznesowego
- ✅ System design ML - architektura, monitoring, deployment
- ✅ Pytania rekrutacyjne - top 20-30 pytań z AI/ML

### PRIORYTET 3: Projekt Nokia (28 stycznia)
**Co przygotować:**
- Opowiedzieć o obecnym projekcie (Anomalib + oscyloskopy)
- Plan rozwoju: MLflow, data pipelines, CI/CD
- Pokazać zrozumienie produkcyjności ML

### PRIORYTET 4: Kurs ML in Production (opcjonalnie)
**Decyzja:** NIE teraz - skup się na projekcie demo i pytaniach

---

## 📋 Demo Script - Co pokazać na rozmowie

### 1. Wprowadzenie (2 min)
"Zbudowałem end-to-end system decyzyjny dla banku, który przewiduje ryzyko kredytowe. 
System składa się z:
- Pipeline treningowy z MLflow tracking
- FastAPI inference service
- Streamlit UI do demo
- Monitoring driftu (PSI/KS)
- RAG assistant do dokumentacji"

### 2. MLflow - Model Registry (3 min)
**Pokaż:**
- MLflow UI z runami
- Metryki: ROC AUC, PR AUC, cost function
- Artefakty: threshold tables, feature importance, drift reference
- Model Registry: Staging/Production

**Mów:**
- "Używam MLflow do trackowania eksperymentów"
- "Każdy run loguje parametry, metryki i artefakty"
- "Model Registry pozwala na promocję modeli do Production"
- "Automatyczny wybór najlepszego modelu na podstawie cost function"

### 3. Overfitting Story (3 min) ⭐ KLUCZOWE
**Pokaż:**
- CV results (train vs test gap)
- Validation set do threshold selection
- Test set tylko do finalnej ewaluacji

**Mów:**
- "Używam 3-way split: train/val/test"
- "Threshold wybieram na validation, nie na test"
- "Cross-validation pokazuje stabilność modelu"
- "Regularizacja: early stopping w XGBoost, class_weight w RF"
- "Metryki na test set są konserwatywne - nie było leakage"

**Pytania które mogą zadać:**
- "Jak unikasz overfittingu?" → CV, validation set, early stopping, regularization
- "Jak wykrywasz overfitting?" → Gap między train/test metrykami
- "Co to leakage?" → Target leakage, data leakage, temporal leakage
- "Jak wybierasz threshold?" → Cost function na validation set

### 4. FastAPI - Inference Service (2 min)
**Pokaż:**
- `/health` endpoint - status modelu
- `/predict` endpoint - prediction + explainability
- Reason codes (SHAP-like dla różnych modeli)

**Mów:**
- "FastAPI służy do inference w produkcji"
- "Model jest cache'owany lokalnie po pierwszym pobraniu"
- "Explainability: reason codes pokazują top features"
- "Request logging do monitoringu"

### 5. Monitoring & Drift (2 min)
**Pokaż:**
- Request logs
- Drift dashboard (PSI/KS)
- Reference distribution vs current

**Mów:**
- "Monitoring driftu: PSI dla kategorycznych, KS dla numerycznych"
- "Reference distribution z train set"
- "Alerty gdy drift przekracza próg"
- "To pozwala wykryć zmiany w danych produkcyjnych"

### 6. RAG Assistant (1 min)
**Pokaż:**
- RAG tab w Streamlit
- Pytanie: "How does threshold work?"
- Citations z dokumentacji

**Mów:**
- "RAG łączy dokumentację z artefaktami MLflow"
- "Extractive answers z citations"
- "Może odpowiadać na pytania o model, metryki, threshold"

### 7. System Design (2 min)
**Mów o architekturze:**
- Training pipeline → MLflow
- Model Registry → Staging/Production
- Inference API → FastAPI
- Monitoring → Request logs + drift
- UI → Streamlit (demo)

**Pytania które mogą zadać:**
- "Jak deployujesz model?" → Model Registry, versioning, A/B testing
- "Jak monitorujesz model w produkcji?" → Drift, request logs, metryki biznesowe
- "Co gdy model się psuje?" → Rollback przez Registry, alerty, human-in-the-loop

---

## 🎓 Top Pytania Rekrutacyjne - AI/ML

### Overfitting & Validation
**Q: Jak unikasz overfittingu?**
**A:**
- Cross-validation do oceny stabilności
- Validation set do hyperparameter tuning
- Test set tylko do finalnej ewaluacji
- Regularizacja (L1/L2, early stopping, dropout)
- Feature selection
- Monitoring gap train/test

**Q: Co to data leakage?**
**A:**
- Target leakage: feature zawiera informację o target (np. "default_date" w credit risk)
- Train/test leakage: informacja z przyszłości w train (temporal leakage)
- Preprocessing leakage: fit na całym dataset zamiast tylko train
- **Przykład:** Scaler.fit() na całym dataset → leakage

**Q: Jak wybierasz threshold?**
**A:**
- Cost function: FP cost vs FN cost
- Business metrics: approval rate, profit
- ROC curve: wybór punktu na krzywej
- Precision/Recall trade-off
- **W projekcie:** Threshold na validation set, cost function FP=1, FN=5

### Problem Framing
**Q: Jak podejść do problemu ML w banku?**
**A:**
1. **Zrozum problem biznesowy:**
   - Jaki problem rozwiązujemy? (credit risk, fraud detection)
   - Jakie są koszty błędów? (FP vs FN)
   - Jakie są constraints? (latency, interpretability)

2. **Zdefiniuj success metrics:**
   - Business metrics: profit, approval rate
   - ML metrics: AUC, precision, recall
   - Operational: latency, throughput

3. **Zbierz dane:**
   - Historical data
   - Label quality
   - Data availability

4. **Baseline:**
   - Simple model (logistic regression)
   - Business rules
   - Existing system

5. **Iterate:**
   - Feature engineering
   - Model selection
   - Hyperparameter tuning
   - Validation

**Q: Jak mierzysz sukces modelu?**
**A:**
- Business metrics: profit, revenue, cost savings
- ML metrics: AUC, precision, recall, F1
- Operational: latency, throughput, availability
- **Ważne:** ML metrics ≠ business value

### System Design
**Q: Jak deployujesz model do produkcji?**
**A:**
- Model Registry (MLflow): versioning, staging/production
- CI/CD pipeline: test → staging → production
- A/B testing: gradual rollout
- Monitoring: drift, performance, errors
- Rollback strategy

**Q: Jak monitorujesz model w produkcji?**
**A:**
- **Data drift:** PSI, KS test
- **Concept drift:** performance degradation
- **Model performance:** accuracy, latency
- **Business metrics:** approval rate, profit
- **Alerts:** automatyczne alerty przy drift

**Q: Co gdy model się psuje w produkcji?**
**A:**
- **Detection:** monitoring alerts
- **Root cause:** drift analysis, error analysis
- **Mitigation:** rollback, retrain, feature fixes
- **Prevention:** better validation, monitoring

### MLflow & MLOps
**Q: Po co MLflow?**
**A:**
- Experiment tracking: parametry, metryki, artefakty
- Model Registry: versioning, staging/production
- Reproducibility: environment, code, data
- Collaboration: shared experiments
- **W projekcie:** tracking runów, model registry, artefakty

**Q: Jak organizujesz eksperymenty?**
**A:**
- Experiments per project/problem
- Run names: descriptive (model-type-hyperparams)
- Tags: project, status, notes
- Artifacts: models, plots, data samples
- **W projekcie:** experiment "bank-credit-risk-http", run names "rf-baseline", "xgb-baseline"

### Feature Engineering
**Q: Jak wybierasz features?**
**A:**
- Domain knowledge: co wpływa na target?
- Feature importance: tree-based models
- Correlation analysis
- Univariate analysis
- **W projekcie:** credit history, amount, age, employment status

**Q: Jak obsługujesz missing values?**
**A:**
- Numeric: median, mean, mode
- Categorical: mode, "missing" category
- Advanced: imputation models
- **W projekcie:** SimpleImputer (median dla numeric, most_frequent dla categorical)

### Model Selection
**Q: Kiedy użyć XGBoost vs Logistic Regression?**
**A:**
- **Logistic Regression:**
  - Interpretability ważna
  - Linear relationships
  - Small dataset
  - Baseline model
  
- **XGBoost:**
  - Non-linear relationships
  - Large dataset
  - Feature interactions
  - Best performance

**Q: Jak wybierasz model?**
**A:**
- Baseline: logistic regression
- Try multiple: RF, XGBoost, Neural Networks
- Compare: CV metrics, validation metrics
- Consider: interpretability, latency, complexity
- **W projekcie:** Porównuję RF, XGBoost, Logistic Regression

---

## 🎯 Plan działania (26-29 stycznia)

### 26 stycznia (dziś) - 4h
- ✅ Napraw projekt (DONE)
- ⏳ Przetestuj pełny flow
- ⏳ Przygotuj demo script
- ⏳ Przećwicz prezentację projektu (2x)

### 27 stycznia - 6h
- ⏳ Przeczytaj top pytania rekrutacyjne (2h)
- ⏳ Przygotuj odpowiedzi na kluczowe pytania (2h)
- ⏳ Przećwicz prezentację projektu (1h)
- ⏳ Przegląd projektu Nokia (1h)

### 28 stycznia - 4h
- ⏳ Rozmowa Nokia (rano)
- ⏳ Finalne przygotowanie do banków (2h)
- ⏳ Przećwicz demo (1h)

### 29 stycznia - rozmowy
- ⏳ Santander (rano)
- ⏳ VeloBank (po południu)

---

## 💡 Kluczowe punkty do zapamiętania

### O projekcie:
1. **Overfitting:** CV, validation set, early stopping, regularization
2. **Problem framing:** Business problem → metrics → data → baseline → iterate
3. **System design:** Training → Registry → Inference → Monitoring
4. **MLflow:** Experiment tracking, Model Registry, artifacts
5. **Monitoring:** Drift (PSI/KS), request logs, performance metrics

### O sobie:
- "Uczę się najlepiej robiąc projekty praktyczne"
- "Mam doświadczenie z Computer Vision (Nokia), teraz rozwijam klasyczny ML"
- "Interesuje mnie produkcyjność ML - MLOps, monitoring, deployment"
- "Chcę pracować nad realnymi problemami biznesowymi"

### O bankach:
- "Rozumiem specyfikę bankową: regulacje, interpretability, risk management"
- "Wiem jak ważne są cost functions i business metrics"
- "Rozumiem potrzebę monitoring i drift detection"

---

## 🚨 Pułapki i podchwytliwe pytania

### "Jak unikasz overfittingu?"
**Pułapka:** Mówienie tylko o technikach, bez pokazania w praktyce
**Dobra odpowiedź:** "W moim projekcie używam CV do oceny stabilności, validation set do threshold selection, early stopping w XGBoost. Pokazuję gap train/test - jeśli jest duży, to overfitting."

### "Co gdy model ma 99% accuracy?"
**Pułapka:** Cieszenie się z wysokiej accuracy
**Dobra odpowiedź:** "Sprawdzam czy to nie leakage, czy dataset nie jest zbalansowany, czy metryka jest odpowiednia. W credit risk ważniejsza jest cost function niż accuracy."

### "Jak deployujesz model?"
**Pułapka:** Mówienie tylko o technologii
**Dobra odpowiedź:** "Używam MLflow Model Registry - model przechodzi przez Staging do Production. Mam monitoring driftu, request logging, rollback strategy."

### "Co gdy model się psuje w produkcji?"
**Pułapka:** Panika, brak planu
**Dobra odpowiedź:** "Monitoring wykrywa problem (drift, performance drop). Analizuję root cause, rollback do poprzedniej wersji, retrain jeśli potrzeba."

---

## 📚 Materiały do szybkiego przeglądu

### Overfitting:
- Train/validation/test split
- Cross-validation
- Regularization (L1/L2, early stopping)
- Feature selection
- Monitoring gap

### Problem Framing:
- Business problem → ML problem
- Success metrics (business + ML)
- Baseline approach
- Iterative improvement

### System Design:
- Training pipeline
- Model Registry
- Inference service
- Monitoring & alerts
- Rollback strategy

### MLflow:
- Experiment tracking
- Model Registry
- Artifacts
- Reproducibility

---

## 🎬 Dwa scenariusze demo — co wybrać

### Opcja A: MVP (`mvp/`) — najbezpieczniejsza
- **Kiedy:** Chcesz zero stresu, wszystko działa w 3 komendach, bez MLflow/docker.
- **Przed rozmową:** `cd mvp && pip install -r requirements.txt && python train.py` (jednorazowo).
- **Na rozmowie:** `uvicorn api:app --port 8000` (term 1), `streamlit run streamlit_app.py` (term 2) → Health, Schema, Predict.
- **Co powiedzieć:** „To uproszczony MVP: pipeline (preprocessing + model), train/val/test, threshold z cost function, FastAPI z schema. W pełnym projekcie dodaję MLflow, drift, RAG.”

### Opcja B: Pełny projekt (MLflow + API + Streamlit)
- **Kiedy:** Chcesz pokazać MLflow, drift, RAG i Model Registry.
- **Przed rozmową (obowiązkowo):**
  1. Uruchom MLflow: `mlflow server --host 0.0.0.0 --port 5050 --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root ./mlartifacts`
  2. Wytrenuj 2–3 modele (żeby w UI były runy):  
     `python -m src.train --model rf --run-name rf-baseline`  
     `python -m src.train --model xgb --run-name xgb-baseline`  
     (opcjonalnie) `python -m src.train --model logreg --run-name logreg-baseline`
  3. Wybierz „best”: `python src/scripts/select_best_model.py`
  4. Sprawdź, że API startuje i odpowiada: `uvicorn src.api.app:app --port 8000` → Health 200, Schema 200.
- **Na rozmowie:** Pokaż MLflow UI (runy, metryki, artefakty), potem API + Streamlit. **Nie uruchamiaj treningów na żywo** — tylko gotowy flow.

### Treningi różnych modeli — tak, ale przed rozmową
- **Tak,** warto mieć w MLflow kilka runów (RF, XGB, ewentualnie LogReg), żeby na rozmowie pokazać eksperymenty, wybór „best” i Model Registry.
- **Nie** uruchamiaj treningów w trakcie demo — trwają minuty, ryzyko błędów sieci/czasu. Zrób to raz wieczorem przed rozmową i przetestuj cały flow.

---

## ✅ Checklist przed rozmową

- [ ] **Ścieżka wybrana:** MVP lub pełny projekt (i przetestowana!)
- [ ] Projekt działa lokalnie (przetestuj!)
- [ ] Przy pełnym projekcie: MLflow + 2–3 treningi + `select_best_model` wykonane wcześniej
- [ ] Demo script przygotowany
- [ ] Przećwiczona prezentacja (5-10 min)
- [ ] Odpowiedzi na top pytania przygotowane
- [ ] Zrozumienie overfitting story w projekcie
- [ ] Zrozumienie system design
- [ ] Zrozumienie problem framing
- [ ] Przygotowane pytania do rekrutera

---

**Powodzenia! 🚀**
