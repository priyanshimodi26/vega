import sys
import argparse
from pathlib import Path

# ── Make sure models/ is importable ──────────────────────────────
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.finbert_scorer      import score_all_transcripts
from models.guidance_classifier import classify_all_transcripts
from models.risk_flagger        import flag_all_transcripts
from models.narrative_gen       import generate_all_narratives


def run_all():
    """Run all four NLP models in sequence."""
    print("=" * 60)
    print("VEGA — NLP Model Pipeline")
    print("=" * 60)

    print("\n[1/4] FinBERT sentiment scorer...")
    stats = score_all_transcripts()
    print(f"      → scored={stats['scored']} skipped={stats['skipped']} failed={stats['failed']}")

    print("\n[2/4] FinBERT-FLS guidance classifier...")
    stats = classify_all_transcripts()
    print(f"      → scored={stats['scored']} skipped={stats['skipped']} failed={stats['failed']}")

    print("\n[3/4] MiniLM + L-M risk flagger...")
    stats = flag_all_transcripts()
    print(f"      → scored={stats['scored']} skipped={stats['skipped']} failed={stats['failed']}")

    print("\n[4/4] Gemini Flash narrative generator...")
    stats = generate_all_narratives()
    print(f"      → generated={stats['generated']} cached={stats['cached']} failed={stats['failed']}")

    print("\n" + "=" * 60)
    print("All models complete.")
    print("=" * 60)


def run_one(model_name: str):
    """Run a single model by name."""
    model_name = model_name.lower()

    if model_name == "finbert":
        print("[VEGA] Running FinBERT sentiment scorer...")
        score_all_transcripts()

    elif model_name == "guidance":
        print("[VEGA] Running FinBERT-FLS guidance classifier...")
        classify_all_transcripts()

    elif model_name == "risk":
        print("[VEGA] Running MiniLM + L-M risk flagger...")
        flag_all_transcripts()

    elif model_name == "narrative":
        print("[VEGA] Running Gemini Flash narrative generator...")
        generate_all_narratives()

    else:
        print(f"[ERROR] Unknown model: {model_name}")
        print("  Valid options: finbert, guidance, risk, narrative")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VEGA NLP model runner")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run a single model: finbert | guidance | risk | narrative"
    )
    args = parser.parse_args()

    if args.model:
        run_one(args.model)
    else:
        run_all()