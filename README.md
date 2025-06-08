# 🏛️ Ramayana Fact Checker (IYD Hackathon)

## What is this project?

**Ramayana Fact Checker** is an offline, open-source, Retrieval-Augmented Generation (RAG) system powered by Ollama and Llama 3.8B. It verifies factual claims about Valmiki's Ramayana using only the original text corpus—no internet, no cloud, just pure local AI.

- **Model:** `ramayana-fact-checker:latest` (Ollama custom model, based on Llama 3.8B)
- **Corpus:** 6 books of Valmiki Ramayana, chunked and indexed for fast retrieval
- **Pipeline:** NLTK keyword extraction → smart chunk retrieval → context-rich prompt → LLM verdict

---

## 🚀 What can it do?

- **Fact-check any statement** about the Ramayana (TRUE / FALSE / INSUFFICIENT_DATA)
- **Explain its verdict** with evidence and reasoning from the actual text
- **Work 100% offline** (no OpenAI API, no cloud, no privacy risk)
- **Process batches of claims** and report accuracy, speed, and confidence

---

## 📊 Metrics (on validation set)

- **Accuracy:** ~90% on 20 diverse Ramayana statements
- **Average time per claim:** 25–35 seconds (on consumer CPU, Llama 3.8B)
- **Total time for 20 claims:** ~9 minutes
- **Memory usage:** 8–16GB RAM (depends on Ollama/Llama model)
- **Explainability:** Every verdict comes with context evidence and reasoning

---

## 🛠️ How to use

### 1. **Install Ollama and the model**

- [Download Ollama](https://ollama.ai) and install for your OS
- Pull the base model:
  ```
  ollama pull llama3:8b
  ```
- Build the custom model (if not already present):
  ```
  ollama create ramayana-fact-checker:latest -f Modelfile.ramayana-fact-checker
  ```

### 2. **Install Python dependencies**

```bash
pip install -r requirements.txt
# Also install wkhtmltoimage for PDF code highlighting (see requirements.txt for details)
```

### 3. **Run a batch evaluation**

```bash
python model_report.py --eval_data_file "data/valmiki_verses_initial_data.jsonl"
```
- See `evaluation_results.jsonl` for detailed verdicts and timings.
- See `evaluation_summary.txt` for overall accuracy and speed.

### 4. **Check a single claim interactively**

```python
from src.ollama_fact_checker import OllamaRamayanaFactChecker
checker = OllamaRamayanaFactChecker()
result = checker.fact_check("Rama is the eldest son of King Dasharatha.")
print(result)
```

---

## 🧠 How does it work?

- **Keyword Extraction:** Uses NLTK to extract key terms from the claim.
- **Chunk Retrieval:** Finds the most relevant Ramayana text chunks using keywords and LLM-generated queries.
- **Prompt Engineering:** Assembles a context-rich prompt with strict fact-checking rules.
- **LLM Verdict:** Llama 3.8B (via Ollama) analyzes the claim and context, outputs verdict, confidence, evidence, and reasoning.
- **Evaluation:** Scripts log per-claim verdicts, confidence, and timing for full transparency.

---

## 📈 Example Output

```json
{
  "claim": "Rama is the eldest son of King Dasharatha.",
  "expected_label": "TRUE",
  "model_verdict": "TRUE",
  "model_confidence": 0.97,
  "model_reasoning": "The context confirms Rama is Dasharatha's eldest son.",
  "evidence": "...",
  "time_taken_sec": 23.4
}
```

---

## 💡 Why is this cool?

- **No hallucinations:** Only facts present in the Ramayana are considered.
- **Explainable AI:** Every answer is backed by evidence and reasoning.
- **Offline & Private:** No data leaves your machine.
- **Hackable:** All code and data are yours—tweak, retrain, or extend as you wish.

---

## 🏁 Quickstart

1. Install Ollama and the model.
2. `pip install -r requirements.txt`
3. Run `python model_report.py --eval_data_file data/valmiki_verses_initial_data.jsonl`
4. See your results and accuracy!

---

## 📂 Codebase

- [`src/ollama_fact_checker.py`](./src/ollama_fact_checker.py) – Core model code for fact checking using custom LLaMA (RAG).
- [`data/`](./data/) – Contains all six Kandas' extracted text and result of running on 20 example statements.
- `model_report.py` — Batch evaluation and metrics
- `docs/generate_report_v2.py` — PDF report generator
- `data/valmiki_verses_initial_data.jsonl` — Example evaluation data

---

## 🤝 Credits

Built for the IYD Hackathon by Team IYD.  
Powered by [Ollama](https://ollama.ai), [Llama 3.8B](https://llama.meta.com/), and open-source Python.

---
