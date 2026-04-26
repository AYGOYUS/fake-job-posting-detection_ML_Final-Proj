import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_dataset, summarize
from text_pipeline import build_text_features
from metadata_pipeline import build_metadata_features
from model_text import train_text_models
from model_metadata import train_metadata_models
from model_combined import train_combined_models
from evaluation import run_evaluation


def main():
    t0 = time.time()

    print("-" * 50)
    print("FAKE JOB POSTING DETECTION")
    print("-" * 50)

    #Load the data and print a quick summary
    print("\n[1/6] Loading data...")
    df = load_dataset()
    summarize(df)

    print("\n[2/6] Building text features...")
    text_X, _, y = build_text_features(df)

    print("\n[3/6] Building metadata features...")
    meta_X, meta_names, _ = build_metadata_features(df)

    print("\n[4/6] Training text models...")
    _, best_text = train_text_models(text_X, y)

    print("\n[5/6] Training metadata models...")
    _, best_meta = train_metadata_models(meta_X, y, meta_names)

    #Combined models + ablation + plots
    print("\n[6/6] Combined model + evaluation...")
    _, best_combined = train_combined_models(text_X, meta_X, y)
    run_evaluation(text_X, meta_X, y)

    print("\n" + "=" * 50)
    print(f"Done in {time.time() - t0:.1f} seconds")
    print(f"  Best text:     {best_text}")
    print(f"  Best metadata: {best_meta}")
    print(f"  Best combined: {best_combined}")
    print("=" * 50)


if __name__ == "__main__":
    main()
