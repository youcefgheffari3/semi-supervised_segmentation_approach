import os
import subprocess

def run_script(script_path):
    """Utility to run a Python script."""
    print(f"\n🚀 Running: {script_path}\n{'-' * 50}")
    result = subprocess.run(["python", script_path])
    if result.returncode != 0:
        print(f"\n❌ Failed: {script_path}")
        exit(1)
    print(f"\n✅ Finished: {script_path}\n{'=' * 50}")


def main():
    # Paths to your scripts
    scripts = [
        "train/train_ancillary.py",
        "train/generate_logits.py",
        "train/train_primary.py",
        "evaluate/evaluate_primary.py"
    ]

    # Run each step
    for script in scripts:
        run_script(script)

    print("\n🎉 Full Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()
