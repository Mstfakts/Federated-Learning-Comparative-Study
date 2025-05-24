import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

# Give a result file
FILE_NAME = "results_2025-05-24_03-22-28.txt"
here = Path(__file__).resolve()
project_root = here.parents[2]
results_dir = project_root / "results" / "fairness_experiments" / FILE_NAME

# Prepare dictionaries to accumulate sums and counts for months 0–4
month_sums = defaultdict(float)
month_counts = defaultdict(int)

# Read all lines from the results file
with open(results_dir, "r") as f:
    lines = f.readlines()

i = 0
# Loop through lines to find the "Class equal.opportunity.difference" blocks
while i < len(lines):
    line = lines[i].strip()
    # Identify the start of the equal opportunity difference block
    if line.startswith("Class equal.opportunity.difference"):
        i += 1
        # Continue reading until we no longer see lines beginning with a month index 0–4
        while i < len(lines) and re.match(r'^\s*[0-4]:', lines[i]):
            match = re.match(r'^\s*([0-4]):\s*([0-9.]+)', lines[i])
            if match:
                month = int(match.group(1))
                value = float(match.group(2))
                month_sums[month] += value
                month_counts[month] += 1
            i += 1
    else:
        i += 1

# Print average difference for each month 0–4
print("Month  Average Opportunity Difference")
for month in range(5):
    if month_counts[month] > 0:
        avg = month_sums[month] / month_counts[month]
        print(f"{month:>2}      {avg:.6f}")
    else:
        print(f"{month:>2}      No data")

# Now extract all overall EOD values and plot them per experiment
with open(results_dir, "r") as f:
    text = f.read()

# Find all occurrences of the overall EqualOpportunityDifference values
eods = [float(val) for val in re.findall(r"EqualOpportunityDifference:\s*([0-9.]+)", text)]

# Compute the average overall EOD
avg_eod = sum(eods) / len(eods)
print(f"\nAverage Equal Opportunity Difference across all experiments: {avg_eod:.4f}")

# Plot EOD per experiment
plt.figure()
plt.plot(range(1, len(eods) + 1), eods, marker="o")
plt.axhline(avg_eod, color="red", linestyle="--", label=f"Overall Avg = {avg_eod:.4f}")
plt.xlabel("Experiment #")
plt.ylabel("Equal Opportunity Difference")
plt.title("EOD per Experiment")
plt.xticks(range(1, len(eods) + 1))
plt.legend()
plt.tight_layout()
plt.show()
