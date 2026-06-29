import json


def load_metrics(metrics_file):

    rows = []

    with open(metrics_file, "r") as f:
        for line in f:

            line = line.strip()

            if line:

                rows.append(json.loads(line))

    return rows