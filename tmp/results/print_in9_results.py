import re
import numpy as np

# Raw data (as a multiline string)
data = """
**results for flacb with seed 0**
original: 97.97530864197532
mixed_rand: 86.37037037037038
mixed_next: 84.34567901234567
mixed_same: 94.19753086419753
no_fg: 61.25925925925926
only_bg_b: 29.259259259259256
only_bg_t: 39.901234567901234
only_fg: 91.90123456790124
**results for flacb with seed 1**
original: 97.90123456790123
mixed_rand: 87.1358024691358
mixed_next: 85.1358024691358
mixed_same: 93.87654320987654
no_fg: 57.99999999999999
only_bg_b: 29.58024691358025
only_bg_t: 39.407407407407405
only_fg: 93.30864197530863
**results for flacb with seed 2**
original: 97.80246913580247
mixed_rand: 86.34567901234568
mixed_next: 84.32098765432099
mixed_same: 94.0246913580247
no_fg: 60.24691358024692
only_bg_b: 30.296296296296298
only_bg_t: 41.82716049382716
only_fg: 92.93827160493827
**results for softcon with seed 0**
original: 52.22222222222223
mixed_rand: 33.53086419753086
mixed_next: 29.950617283950614
mixed_same: 49.25925925925926
no_fg: 30.987654320987655
only_bg_b: 21.975308641975307
only_bg_t: 25.333333333333336
only_fg: 39.382716049382715
**results for softcon with seed 1**
original: 47.80246913580247
mixed_rand: 30.864197530864196
mixed_next: 28.790123456790123
mixed_same: 47.28395061728395
no_fg: 24.88888888888889
only_bg_b: 17.308641975308642
only_bg_t: 25.62962962962963
only_fg: 30.864197530864196
**results for softcon with seed 2**
original: 42.39506172839506
mixed_rand: 27.209876543209877
mixed_next: 25.703703703703706
mixed_same: 39.925925925925924
no_fg: 28.197530864197528
only_bg_b: 20.271604938271608
only_bg_t: 21.23456790123457
only_fg: 28.74074074074074
**results for sd with seed 0**
original: 98.22222222222223
mixed_rand: 89.70370370370371
mixed_next: 88.04938271604938
mixed_same: 94.71604938271605
no_fg: 61.432098765432094
only_bg_b: 28.469135802469136
only_bg_t: 38.69135802469136
only_fg: 93.85185185185185
**results for sd with seed 1**
original: 98.09876543209877
mixed_rand: 88.22222222222223
mixed_next: 86.93827160493827
mixed_same: 95.38271604938272
no_fg: 63.48148148148148
only_bg_b: 34.32098765432099
only_bg_t: 44.24691358024692
only_fg: 94.34567901234568
**results for sd with seed 2**
original: 98.14814814814815
mixed_rand: 88.8395061728395
mixed_next: 87.67901234567901
mixed_same: 94.22222222222221
no_fg: 62.88888888888889
only_bg_b: 31.48148148148148
only_bg_t: 39.50617283950617
only_fg: 92.93827160493827
"""

# Parse the data
results = {}
current_dataset = None
for line in data.split("\n"):
    match = re.match(r"\*\*results for (.+?) with seed (\d+)\*\*", line)
    if match:
        current_dataset = match.group(1)
        if current_dataset not in results:
            results[current_dataset] = {}
    elif current_dataset and ":" in line:
        method, value = line.split(":")
        method = method.strip()
        value = float(value.strip())
        if method not in results[current_dataset]:
            results[current_dataset][method] = []
        results[current_dataset][method].append(value)

# Compute mean and standard deviation
for dataset, methods in results.items():
    print(f"Performance for {dataset}:")
    for method, values in methods.items():
        mean = np.mean(values)
        std = np.std(values, ddof=1)  # Sample standard deviation
        print(f"  {method}: {mean:.2f} ± {std:.2f}")
    print()
