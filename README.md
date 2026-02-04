# Case-law Citation Helper via Model Merging (LoRA Experts + TIES/DARE)

This repository contains a Jupyter/Colab notebook (`LLM_project.ipynb`) that demonstrates an end-to-end workflow to build a **Case-law Citation Helper** by **fine-tuning two specialized LoRA “experts”** and then **merging** them into a single adapter using **TIES** and **DARE**-style strategies.

Instead of expensive multi-task full fine-tuning, the notebook relies on **Parameter-Efficient Fine-Tuning (PEFT)** with **LoRA** and performs merging **directly on the adapters**.

**Base model:** `unsloth/meta-llama-3.1-8b-instruct-bnb-4bit` (4-bit quantized)  
**Framework:** Unsloth + Hugging Face ecosystem (Transformers/TRL/PEFT)

---

## 🛠️ What the notebook does

### 1) Environment setup (Colab-ready)
- Installs **Unsloth** and compatible dependencies (`xformers`, `trl`, `peft`, `accelerate`, `bitsandbytes`).
- Mounts Google Drive and uses a `BASE_PATH` to persist checkpoints, adapters, and evaluation artifacts.

### 2) Model initialization
- Loads **Llama 3.1 8B Instruct** in **4-bit** to reduce VRAM usage.
- Configures LoRA with a lightweight adapter (rank/alpha/dropout) to enable later merging.

Key defaults in the notebook:
- `max_seq_length = 1024`
- `load_in_4bit = True`
- LoRA: `r = 16`, `alpha = 32`, `dropout = 0.05`

### 3) Expert 1 fine-tuning (CaseHOLD)
Goal: specialize the model for **holding selection** (multiple-choice QA in case-law settings).

Pipeline:
- Loads and cleans legal text (normalization, noise removal, consistent formatting).
- Formats each example into a structured **A–E multiple-choice** prompt using the Llama chat template.
- Fine-tunes a LoRA adapter (SFT) and saves **only** the adapter weights to Drive.
- Evaluates on a validation split using **accuracy**:
  - evaluates Expert 1 adapter
  - evaluates the base model by temporarily disabling the adapter
  - saves results as JSON and prints a comparison table

### 4) Expert 2 fine-tuning (LegalBench-Instruct)
Goal: learn **broad legal reasoning and instruction-following** across heterogeneous legal tasks.

Pipeline:
- Loads `Equall/legalbench_instruct` and performs a **stratified, disjoint split** (train/eval/test-like structure).
- Fine-tunes a second LoRA adapter (SFT) and saves the adapter to Drive.
- Evaluates with a *task-agnostic* suite of metrics:
  - **Strict Exact Match (Strict EM)**
  - **Contains Ground Truth (Contains)**
  - **Label Accuracy (Label Acc)** for label-like answers (yes/no, single letters, ordinals)
  - **ROUGE-L F1** for free-form (non-label) answers  
- Saves metrics as JSON and prints a comparative table.

### 5) Adapter merging (TIES / DARE / DARE+TIES)
After training the two experts, the notebook merges them into a **single LoRA adapter**.

Supported merge modes:
- `ties` : Trim → Elect Sign → Merge
- `dare` : Drop + (optional) Rescale on deltas, then weighted averaging
- `dare_ties` : DARE preprocessing followed by TIES merging

The notebook performs a **small hyperparameter sweep**, producing multiple merged adapters and saving them to separate folders alongside a manifest.


### 6) Evaluation of merged adapters + final test evaluation
- Evaluates **all merged adapters** on both CaseHOLD and LegalBench evaluation settings.
- Selects the best candidates and runs a **final evaluation on hold-out test splits**.
- Persists outputs (JSON/CSV) to Drive for reproducibility.
- Includes a final qualitative section and a references section.

---

## 📂 Datasets used
- **CaseHOLD**: multiple-choice holding selection from judicial contexts (A–E options).
- **LegalBench-Instruct** (`Equall/legalbench_instruct`): a broad benchmark covering many legal reasoning tasks.

> Note: the notebook downloads/loads datasets through the Hugging Face `datasets` library.

## 🚀 How to run

### Recommended: Google Colab (GPU)
1. Open `LLM_project.ipynb` in Colab.
2. Ensure you have a GPU runtime enabled.
3. Run the setup cells (installs + Drive mount).
4. Update path configuration:
   ```python
   BASE_PATH = "/content/drive/Shareddrives/MIRCV Project/LLM/"

## 📊 Outputs

The notebook produces:
- LoRA adapters for Expert 1 and Expert 2 (adapter_model.safetensors + config)
- Multiple merged LoRA adapters (one per sweep configuration)
- Evaluation reports (JSON) and summary tables
- Final test evaluation artifacts (JSON/CSV)


## 👥 Authors

M. Fabiani

T. Falaschi

A. Franchini

R. A. Sacco
