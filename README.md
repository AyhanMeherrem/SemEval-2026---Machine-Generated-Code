# SemEval 2026 Task 13 (Task B): AI Code Detection Solution

This repository contains my solution for **SemEval 2026 Task 13: Subtask B**, focusing on classifying code as Human-written or AI-generated (across various models like DeepSeek, Qwen, Llama, etc.).

The solution implements a **Two-Stage Pipeline** using `microsoft/unixcoder-base` to handle extreme class imbalance and hard-to-distinguish examples.

## 🚀 Approach Overview

The solution is split into two notebooks representing a progressive training strategy:

### 1. Phase 1: Mining Hard Negatives (`SemEval_13TaskB_Phase1.ipynb`)
* **Initial Training:** Trains a baseline UniXcoder model using **Focal Loss** on a pruned, balanced dataset (100k Humans + All AI).
* **Hard Example Mining:** After training, the model predicts on the *entire* 500k dataset.
* **Loss Analysis:** We identify the "Hard Negatives" (samples with the highest loss) where the model is confused. These indices are saved to `hard_negative_indices.json` for the next phase.

### 2. Phase 2: Grandmaster Training & Stabilization (`SemEval_13TaskB_Phase2.ipynb`)
* **3-Tier Weighted Sampling:** A fresh model is trained using a **WeightedRandomSampler** that prioritizes data based on difficulty:
    * 🔴 **Tier 1 (5.0x weight):** Hard Examples (mined from Phase 1).
    * 🟡 **Tier 2 (3.0x weight):** Easy AI examples (Hidden gems).
    * 🟢 **Tier 3 (1.0x weight):** Easy Human examples (Background noise).
* **Grandmaster Training:** Trains for 5 epochs with high batch sizes to master the difficult patterns.
* **Cool-Down Phase (Stabilization):** A final 1-epoch training run on the *natural* distribution (no sampler) with a very low learning rate (`5e-6`). This corrects "model paranoia" and significantly improves recall for the Human class.

## 🛠️ Key Techniques
* **Model:** `microsoft/unixcoder-base`
* **Loss Function:** Focal Loss (to handle class imbalance).
* **Sampling:** Weighted Random Sampler with dynamic priorities.
* **Precision:** Mixed Precision (FP16) for efficiency.

## 📂 File Structure
* `SemEval_13TaskB_Phase1.ipynb`: Baseline training and extraction of hard examples.
* `SemEval_13TaskB_Phase2.ipynb`: Advanced weighted training and final stabilization.
* `hard_negative_indices.json`: (Generated output) Indices of the most difficult training samples.
* `submission_phase4.csv`: Final submission file.

## ⚡ How to Run
1.  **Run Phase 1:** Execute the first notebook to train the baseline and generate the `hard_negative_indices.json` file.
2.  **Run Phase 2:** Ensure the JSON file is available, then execute the second notebook to train the final model and generate the submission.
