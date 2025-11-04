# LLM Fine-Tuning & RLHF on Qwen2.5-0.5B

This repository contains two notebooks showing how to adapt a small open LLM using:
1. **Supervised Fine-Tuning (SFT)** with LoRA and 4-bit quantization
2. **RLHF-style optimization** using DPO / GRPO (preference-based methods)

The goal is to demonstrate a lightweight, reproducible pipeline to make an LLM follow instructions better.

---

## 📘 Notebooks

- `notebooks/01_instruction_finetuning_qwen.ipynb`  
  Loads `Qwen/Qwen2.5-0.5B`, quantizes it to 4-bit, attaches LoRA adapters and fine-tunes it on the public `giuliadc/orangesum_5k` dataset (French summarization ➜ instruction format).  
  The notebook also shows how to test the model before/after and how to merge LoRA weights.

- `notebooks/02_rlhf_dpo_grpo_qwen.ipynb`  
  (to be added) Extends the SFT model with **preference optimization** using the 🤗 TRL library: DPO and GRPO.

---

## ⚙️ Install & run

```bash
pip install -r requirements.txt
jupyter lab
