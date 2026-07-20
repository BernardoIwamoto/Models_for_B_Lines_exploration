def get_series(metrics, key):

    x = []
    y = []

    for row in metrics:

        if key in row:

            x.append(row["iteration"])

            y.append(row[key])

    return x, y