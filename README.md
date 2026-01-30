# LLM from Scratch: Building a GPT-style Model

This project is a deep dive into the internal workings of Large Language Models (LLMs). Based on the `llm-from-scratch.ipynb` notebook, it documents the step-by-step journey of building, understanding, and training a GPT-style transformer model from the ground up using PyTorch.

## 🚀 Project Overview

The core objective is to demystify the "black box" of LLMs by implementing every component manually. The project follows a structured progression:

1.  **Working with Text Data**:
    *   Implementing custom regex-based tokenizers.
    *   Handling special tokens like `<|endoftext|>` and `<|unk|>`.
    *   Transitioning to industry-standard Byte Pair Encoding (BPE) using `tiktoken`.
2.  **Dataset Preparation**:
    *   Creating a sliding window mechanism to generate input-target pairs for next-token prediction.
    *   Building custom PyTorch `Dataset` and `DataLoader` for efficient training.
3.  **Embeddings**:
    *   Implementing Token Embeddings to map IDs to continuous vectors.
    *   Adding Positional Embeddings to provide context about token order.
4.  **Attention Mechanisms**:
    *   **Self-Attention**: The fundamental building block.
    *   **Causal Attention**: Implementing masking to prevent the model from "looking into the future" during training.
    *   **Multi-Head Attention**: Allowing the model to attend to different parts of the input sequence simultaneously.
5.  **GPT Architecture**:
    *   Assembling the transformer blocks.
    *   Implementing Layer Normalization, Feed-Forward networks, and Shortcut Connections.

## 🛠️ Setup Instructions

### 1. Environment Setup
Create and activate a virtual environment to manage dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Project
Launch JupyterLab to explore the implementation:
```bash
jupyter lab
```

## 📦 Core Stack
*   **PyTorch**: Deep learning framework for model implementation and training.
*   **Tiktoken**: Fast BPE tokenizer used by OpenAI models (GPT-3.5/GPT-4).
*   **Matplotlib**: For visualizing training loss and attention weights.
*   **NumPy/Pandas**: Data manipulation and analysis.

## 📖 Key Components Implemented
*   `SimpleTokenizerV1` / `V2`: Custom tokenizer implementations.
*   `GPTDatasetV1`: Dataset generator using sliding windows.
*   `CausalAttention`: Scaled dot-product attention with causal masking.
*   `MultiHeadAttention`: Efficient parallel attention heads.

---
*Developed as part of Course 22969 - Build your own LLM.*
