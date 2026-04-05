import os
import subprocess
import sys

def run_step(name, command):
    print(f"\n🚀 Running: {name}")
    print("-" * 50)

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"❌ Error in step: {name}")
        sys.exit(1)

    print(f"✅ Completed: {name}")


def main():
    steps = [
        ("Dataset Analysis", "python -m dataset_analyzer.analyze_dataset"),
        ("Train Lung Model", "python -m training.train_lung"),
        ("Train Infection Model", "python -m training.train_infection"),
        ("Evaluate Model", "python -m evaluation.evaluate"),
        ("Ablation Study", "python -m evaluation.ablation"),
        ("Comparison Graph", "python -m evaluation.compare_models"),
        ("Run Pipeline", "python -m pipeline.run_pipeline"),
    ]

    for name, cmd in steps:
        run_step(name, cmd)

    print("\n🌐 Launching Streamlit UI...")
    subprocess.run("streamlit run app/app.py", shell=True)


if __name__ == "__main__":
    main()