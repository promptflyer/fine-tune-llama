# Fine‑Tuning Llama 3.2 Small with Ollama on Mac M3 Pro

This guide explains how to fine‑tune **Llama 3.2 Small** using your PDF documents on a Mac M3 Pro with **Ollama**. The
goal is to embed your documents directly into the model for **fast responses** without relying on RAG.

---

## 📂 Step 1: Prepare Your Data

1. **Extract text from PDFs**  
   Install libraries:
   ```bash
   pip install PyPDF2 pdfminer.six unstructured
   ```
   Example:
   ```python
   from PyPDF2 import PdfReader

   reader = PdfReader("document.pdf")
   text = ""
   for page in reader.pages:
       text += page.extract_text()
   ```

2. **Convert into Q&A pairs (JSONL format)**  
   Example:
   ```json
   {"instruction": "What is the refund policy?", "response": "Refunds are available within 30 days of purchase."}
   {"instruction": "List the required documents for visa application.", "response": "Passport, application form, and proof of funds."}
   ```

3. **Clean the text**
   - Remove headers/footers/page numbers.
   - Normalize whitespace.
   - Keep responses concise and factual.

---

## 🖥️ Step 2: Install Ollama and Base Model

1. Install Ollama:  
   [https://ollama.ai](https://ollama.ai)

2. Pull the base Llama model:
   ```bash
   ollama pull llama3.2
   ```

3. Test it:
   ```bash
   ollama run llama3.2
   ```

---

## ⚙️ Step 3: Fine‑Tune Externally (LoRA/PEFT)

Ollama doesn’t fine‑tune directly inside the `Modelfile`. Instead:

1. Use Hugging Face `transformers` + `peft` to fine‑tune with LoRA adapters:
   ```bash
   pip install transformers datasets accelerate peft
   ```

2. Train your dataset (JSONL Q&A pairs) with LoRA.  
   This produces adapter weights (`adapter_model.bin`).

---

## 🗂️ Step 4: Import into Ollama

Once you have LoRA adapter weights:

1. Create a `Modelfile`:
   ```text
   FROM llama3.2
   ADAPTER ./adapter_model.bin
   PARAMETER temperature 0.2
   PARAMETER top_p 0.9
   ```

2. Build your custom model:
   ```bash
   ollama create mydocs -f Modelfile
   ```

---

## 🚀 Step 5: Run Your Fine‑Tuned Model

Test it:

```bash
ollama run mydocs
```

Example:

```
> What is the refund policy?
Refunds are available within 30 days of purchase.
```

---

## ⚡ Step 6: Optimize for Speed

- Ollama automatically optimizes for **Apple Silicon (M‑series)**.
- Use **LoRA adapters** for lightweight fine‑tuning.
- Keep prompts short and structured.
- Consider **quantization** for smaller memory footprint.

---

## ⚖️ Trade‑Offs

- **Fine‑tuning**: Faster inference, answers directly from docs. Requires retraining if docs change.
- **RAG**: More flexible, no retraining needed, but slower due to retrieval overhead.

For static knowledge bases (manuals, policies, FAQs), fine‑tuning is ideal.

---

## 🛠️ Example: Convert PDFs to JSONL

```python
import json
from PyPDF2 import PdfReader

reader = PdfReader("document.pdf")
qa_pairs = []

for page in reader.pages:
   text = page.extract_text()
   qa_pairs.append({
      "instruction": "Summarize this page:",
      "response": text.strip()
   })

with open("dataset.jsonl", "w") as f:
   for pair in qa_pairs:
      f.write(json.dumps(pair) + "\n")
```

---

## ✅ Summary

- Extract text from PDFs.
- Convert into JSONL Q&A pairs.
- Fine‑tune with Hugging Face (LoRA).
- Import adapter weights into Ollama via `Modelfile`.
- Run your custom model locally for **fast, document‑aware answers**.

```

---