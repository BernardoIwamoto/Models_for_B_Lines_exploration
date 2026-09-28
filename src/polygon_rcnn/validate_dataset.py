"""Read-only dataset integrity check: python -m src.polygon_rcnn.validate_dataset."""

import argparse
import json

from src.polygon_rcnn.annotation import audit_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data", help="Dataset root with train/val/test")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    report = audit_dataset(args.root)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        with open(args.output, "x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
