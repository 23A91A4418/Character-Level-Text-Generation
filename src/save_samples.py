import json
import subprocess


def run_generation(model, temperature, seed_text="To be or not to be"):
    command = [
        "python",
        "src/generate.py",
        "--model",
        model,
        "--temperature",
        str(temperature),
        "--seed_text",
        seed_text,
        "--length",
        "300"
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    return result.stdout.strip()


def main():
    output = {
        "lstm": {},
        "transformer": {}
    }

    temperatures = [0.5, 1.0, 1.5]

    for model in ["lstm", "transformer"]:
        for temp in temperatures:
            key = f"temperature_{temp}"

            print(f"Generating {model} at temperature {temp}...")

            sample1 = run_generation(model, temp)
            sample2 = run_generation(model, temp)

            output[model][key] = [
                sample1,
                sample2
            ]

    with open(
        "results/generated_samples.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(output, f, indent=4)

    print("generated_samples.json created successfully!")


if __name__ == "__main__":
    main()