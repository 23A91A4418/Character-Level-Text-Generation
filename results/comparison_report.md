# Model Comparison Report

### Perplexity Comparison

| Model | Final Loss | Approx Perplexity |
|---|---:|---:|
| LSTM | 0.7342 | 2.08 |
| Transformer | 0.0335 | 1.03 |

### Qualitative Analysis

Both models were trained on the Tiny Shakespeare dataset for character-level text generation using PyTorch.

The LSTM model generated highly readable and meaningful Shakespeare-style dialogue. It successfully learned speaker names such as MENENIUS, CORIOLANUS, and dialogue formatting with natural sentence flow. The generated output was coherent and stylistically close to the original dataset.

The Transformer model achieved significantly lower loss and better perplexity compared to the LSTM, indicating stronger next-character prediction performance. However, during initial generation it produced repetitive outputs like repeated characters. After improving context handling and adjusting temperature sampling, the generated text became much better and showed recognizable words, sentence structures, and dramatic dialogue patterns.

Temperature had a major effect on both models. Lower temperature (0.5) produced safer and more repetitive text with higher coherence. Standard temperature (1.0) balanced creativity and readability. Higher temperature (1.5) generated more diverse and creative outputs but also increased grammatical mistakes and randomness.

Overall, the LSTM performed better in qualitative text generation quality, while the Transformer performed better quantitatively in terms of training loss and perplexity. This demonstrates the practical tradeoff between generation quality and prediction efficiency.