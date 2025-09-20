import argparse, json, os

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    metrics_path = os.path.join(args.run_dir, "metrics.json")
    with open(metrics_path, "r") as f:
        metrics = json.load(f)

    # Pretty print
    print(json.dumps(metrics, indent=2))
