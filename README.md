# Character-Level Text Generation using LSTM and Transformer

## Project Overview

This project implements and compares two fundamental sequence models for character-level text generation using PyTorch:

- LSTM (Long Short-Term Memory)
- Mini Transformer

Both models were trained on the Tiny Shakespeare dataset to generate Shakespeare-style text one character at a time.

The project includes complete data preparation, model training, text generation, result visualization, Docker containerization, and comparative analysis.

---

## Dataset

Dataset used:

Tiny Shakespeare Dataset

Source:
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

This dataset contains approximately 1 million characters and 65 unique characters, making it ideal for CPU-friendly character-level text generation.

---

## Project Structure

```text
Character-Level-Text-Generation/
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── requirements.txt
├── README.md
├── input/
├── models/
├── results/
└── src/
Models Implemented
LSTM Model

The LSTM model uses:

Embedding Layer
Multi-layer LSTM
Fully Connected Output Layer

It performs strong qualitative text generation and produces highly readable Shakespeare-style dialogue.

Transformer Model

The Transformer model uses:

Token Embedding
Positional Embedding
Transformer Encoder Layers
Fully Connected Output Layer

It performs strong quantitative prediction with lower loss and better perplexity.

Training

Train LSTM:

python src/train.py --model lstm

Train Transformer:

python src/train.py --model transformer
Text Generation

Generate text using LSTM:

python src/generate.py --model lstm

Generate text using Transformer:

python src/generate.py --model transformer
Docker Execution

Build container:

docker-compose build

Run training:

docker-compose run --rm app python src/train.py --model lstm
Results

Generated outputs:

loss_curves.png
generated_samples.json
comparison_report.md

Main findings:

LSTM generated better readable text
Transformer achieved better loss and perplexity
Temperature significantly affected creativity and coherence
Final Comparison
Model	Final Loss	Approx Perplexity
LSTM	0.7342	2.08
Transformer	0.0335	1.03

Transformer performed better quantitatively, while LSTM performed better qualitatively.

Technologies Used
Python
PyTorch
NumPy
Matplotlib
Docker
Docker Compose

---

# Your Task Now

Create these 5 files exactly.

After that, your project is submission-ready.