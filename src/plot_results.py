import json
import matplotlib.pyplot as plt


# Load saved loss histories
with open("models/lstm_loss.json", "r") as f:
    lstm_loss = json.load(f)

with open("models/transformer_loss.json", "r") as f:
    transformer_loss = json.load(f)

epochs = list(range(1, len(lstm_loss) + 1))

plt.figure(figsize=(8, 5))

plt.plot(epochs, lstm_loss, label="LSTM Loss")
plt.plot(epochs, transformer_loss, label="Transformer Loss")

plt.title("Training Loss Comparison")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("results/loss_curves.png")
plt.close()

print("loss_curves.png created successfully!")