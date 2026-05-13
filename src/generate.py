import json
import argparse
import torch
import torch.nn.functional as F

from model_lstm import LSTMModel
from model_transformer import TransformerModel


def load_mappings():
    with open("models/char_to_int.json", "r", encoding="utf-8") as f:
        char_to_int = json.load(f)

    with open("models/int_to_char.json", "r", encoding="utf-8") as f:
        int_to_char = json.load(f)

    # Convert JSON string keys back to integers
    int_to_char = {int(k): v for k, v in int_to_char.items()}

    return char_to_int, int_to_char


def predict_lstm(model, char, hidden, char_to_int, int_to_char, temperature):
    x = torch.tensor([[char_to_int[char]]], dtype=torch.long)

    output, hidden = model(x, hidden)

    output = output / temperature
    probs = F.softmax(output, dim=1)

    top_char = torch.multinomial(probs, 1)[0]
    predicted_char = int_to_char[top_char.item()]

    return predicted_char, hidden


def generate_lstm(model, seed_text, length, temperature):
    char_to_int, int_to_char = load_mappings()

    model.eval()
    hidden = model.init_hidden(1)

    generated = seed_text

    # Warm-up with seed text
    for char in seed_text:
        _, hidden = predict_lstm(
            model,
            char,
            hidden,
            char_to_int,
            int_to_char,
            temperature
        )

    last_char = seed_text[-1]

    for _ in range(length):
        next_char, hidden = predict_lstm(
            model,
            last_char,
            hidden,
            char_to_int,
            int_to_char,
            temperature
        )

        generated += next_char
        last_char = next_char

    return generated


def generate_transformer(model, seed_text, length, temperature):
    char_to_int, int_to_char = load_mappings()

    model.eval()
    generated = seed_text

    for _ in range(length):
        # Take last 100 chars as context
        context = generated[-100:]

        # Pad if smaller than 100
        if len(context) < 100:
            context = " " * (100 - len(context)) + context

        # IMPORTANT: must be outside if block
        input_seq = [
            char_to_int.get(ch, char_to_int[" "])
            for ch in context
        ]

        x = torch.tensor([input_seq], dtype=torch.long)

        with torch.no_grad():
            output = model(x)

        output = output[-1] / temperature
        probs = F.softmax(output, dim=0)

        next_char_idx = torch.multinomial(probs, 1).item()
        next_char = int_to_char[next_char_idx]

        generated += next_char

    return generated


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "transformer"]
    )

    parser.add_argument(
        "--seed_text",
        type=str,
        default="To be or not to be"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7
    )

    parser.add_argument(
        "--length",
        type=int,
        default=500
    )

    args = parser.parse_args()

    char_to_int, _ = load_mappings()
    vocab_size = len(char_to_int)

    if args.model == "lstm":
        model = LSTMModel(vocab_size=vocab_size)

        model.load_state_dict(
            torch.load(
                "models/lstm_model.pth",
                map_location=torch.device("cpu")
            )
        )

        result = generate_lstm(
            model,
            args.seed_text,
            args.length,
            args.temperature
        )

    else:
        model = TransformerModel(vocab_size=vocab_size)

        model.load_state_dict(
            torch.load(
                "models/transformer_model.pth",
                map_location=torch.device("cpu")
            )
        )

        result = generate_transformer(
            model,
            args.seed_text,
            args.length,
            args.temperature
        )

    print("\nGenerated Text:\n")
    print(result)


if __name__ == "__main__":
    main()