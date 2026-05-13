import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_lstm import LSTMModel
from model_transformer import TransformerModel
from prepare_data import (
    load_text,
    create_vocab,
    encode_text,
    create_sequences
)


# -----------------------------
# Settings
# -----------------------------
SEQ_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 0.001
MAX_SAMPLES = 100000


def prepare_dataset():
    print("Preparing dataset...")

    text = load_text()
    _, vocab_size, char_to_int, _ = create_vocab(text)

    encoded_text = encode_text(text, char_to_int)
    X, y = create_sequences(encoded_text, SEQ_LENGTH)

    # Faster CPU training
    X = X[:MAX_SAMPLES]
    y = y[:MAX_SAMPLES]

    X = torch.tensor(X, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X, y)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    print(f"Vocabulary Size: {vocab_size}")
    print(f"Training batches: {len(dataloader)}")

    return dataloader, vocab_size


def train_lstm():
    dataloader, vocab_size = prepare_dataset()

    model = LSTMModel(vocab_size=vocab_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_history = []

    print("\nTraining LSTM...\n")

    for epoch in range(EPOCHS):
        hidden = model.init_hidden(BATCH_SIZE)
        epoch_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):

            if inputs.size(0) != BATCH_SIZE:
                continue

            hidden = tuple([each.data for each in hidden])

            model.zero_grad()

            output, hidden = model(inputs, hidden)

            targets = targets.view(-1)

            loss = criterion(output, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1
            )

            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 200 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(
            f"\nEpoch {epoch+1} Completed "
            f"| Avg Loss: {avg_loss:.4f}\n"
        )

    os.makedirs("models", exist_ok=True)

    torch.save(
        model.state_dict(),
        "models/lstm_model.pth"
    )

    with open(
        "models/lstm_loss.json",
        "w"
    ) as f:
        json.dump(loss_history, f, indent=4)

    print("LSTM training completed successfully!")


def train_transformer():
    dataloader, vocab_size = prepare_dataset()

    model = TransformerModel(vocab_size=vocab_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_history = []

    print("\nTraining Transformer...\n")

    for epoch in range(EPOCHS):
        epoch_loss = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):

            model.zero_grad()

            output = model(inputs)

            targets = targets.view(-1)

            loss = criterion(output, targets)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1
            )

            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 200 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        print(
            f"\nEpoch {epoch+1} Completed "
            f"| Avg Loss: {avg_loss:.4f}\n"
        )

    os.makedirs("models", exist_ok=True)

    torch.save(
        model.state_dict(),
        "models/transformer_model.pth"
    )

    with open(
        "models/transformer_loss.json",
        "w"
    ) as f:
        json.dump(loss_history, f, indent=4)

    print("Transformer training completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["lstm", "transformer"]
    )

    args = parser.parse_args()

    if args.model == "lstm":
        train_lstm()

    elif args.model == "transformer":
        train_transformer()