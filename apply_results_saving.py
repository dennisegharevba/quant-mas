"""
One-time patch: adds a "save full results to CSV" line to each
evaluate_modelN.py script, right after results are computed.
"""

from pathlib import Path

BACKTEST_DIR = Path(__file__).resolve().parent / "backtest_engine"

ANCHOR = "    results = run_all(processed)"

INSERT_TEMPLATE = '''    results = run_all(processed)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    results.to_csv(results_dir / "{filename}", index=False)
'''

FILES = {
    "evaluate_model1.py": "model1_results.csv",
    "evaluate_model2.py": "model2_results.csv",
    "evaluate_model3.py": "model3_results.csv",
    "evaluate_model4.py": "model4_results.csv",
    "evaluate_model5.py": "model5_results.csv",
}

if __name__ == "__main__":
    for filename, results_filename in FILES.items():
        path = BACKTEST_DIR / filename
        content = path.read_text(encoding="utf-8")

        if "results_dir.mkdir" in content:
            print(f"[skip] {filename} already patched")
            continue

        if content.count(ANCHOR) != 1:
            print(f"[FAIL] {filename}: expected exactly 1 occurrence of anchor, found {content.count(ANCHOR)} - not patched, check manually")
            continue

        new_content = content.replace(ANCHOR, INSERT_TEMPLATE.format(filename=results_filename).rstrip("\n"))
        path.write_text(new_content, encoding="utf-8")
        print(f"[OK] patched {filename}")

    print("\nDone. Re-run each evaluate_modelN.py once to populate backtest_engine/results/.")