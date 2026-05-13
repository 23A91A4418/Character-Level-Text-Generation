import os
import json
import numpy as np

# File path
DATA_PATH = "input/shakespeare.txt"

# Sequence length for training
SEQ_LENGTH = 100


def load_text():
    """
    Load the full text dataset
    """
    with open(DATA_PATH, "r", encoding="utf-8") as file:
        text = file.read()
    return text


def create_vocab(text):
    """
    Create character vocabulary and mappings
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

    return chars, vocab_size, char_to_int, int_to_char


def encode_text(text, char_to_int):
    """
    Convert text characters into integer encoding
    """
    encoded = np.array([char_to_int[ch] for ch in text])
    return encoded


def create_sequences(encoded_text, seq_length):
    """
    Create input-target sequences

    Example:
    Input  = first 100 chars
    Target = next 100 chars shifted by 1
    """
    inputs = []
    targets = []

    for i in range(0, len(encoded_text) - seq_length):
        input_seq = encoded_text[i:i + seq_length]
        target_seq = encoded_text[i + 1:i + seq_length + 1]

        inputs.append(input_seq)
        targets.append(target_seq)

    return np.array(inputs), np.array(targets)


def save_mappings(char_to_int, int_to_char):
    """
    Save mappings for generation script
    """
    os.makedirs("models", exist_ok=True)

    with open("models/char_to_int.json", "w", encoding="utf-8") as f:
        json.dump(char_to_int, f, indent=4)

    with open("models/int_to_char.json", "w", encoding="utf-8") as f:
        json.dump(int_to_char, f, indent=4)


def main():
    print("Loading dataset...")

    text = load_text()
    print(f"Total characters in dataset: {len(text)}")

    chars, vocab_size, char_to_int, int_to_char = create_vocab(text)

    print(f"Unique characters (Vocabulary Size): {vocab_size}")

    encoded_text = encode_text(text, char_to_int)

    X, y = create_sequences(encoded_text, SEQ_LENGTH)

    print(f"Total training sequences: {len(X)}")
    print(f"Input shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    save_mappings(char_to_int, int_to_char)

    print("Mappings saved successfully in models/")
    print("Data preparation completed successfully.")


if __name__ == "__main__":
    main()