# from collections import defaultdict
# import numpy as np

# def process_data(input_file, output_file):
#     with open(input_file, "r", encoding="utf-8") as f:
#         lines = f.readlines()
    
#     data = defaultdict(list)
#     current_cut = None
#     results = defaultdict(lambda: defaultdict(list))

#     for line in lines:
#         line = line.strip()
#         if line.startswith("new cut:"):
#             current_cut = int(line.split(":")[-1].strip())
#         elif line:
#             parts = line.split()
#             if len(parts) >= 5:
#                 dataset = " ".join(parts[2:-2])  # Dataset name
#                 reach = float(parts[-2])         # Reach value
#                 time_taken = float(parts[-1])    # Time taken
#                 results[current_cut][dataset].append((reach, time_taken))

#     # Compute averages and write to output
#     with open(output_file, "w", encoding="utf-8") as f:
#         for cut, datasets in results.items():
#             f.write(f"Cut Value: {cut}\n")
#             f.write("Dataset | Avg Reach | Avg Time Taken\n")
#             f.write("-" * 40 + "\n")
#             for dataset, values in datasets.items():
#                 avg_reach = np.mean([v[0] for v in values])
#                 avg_time = np.mean([v[1] for v in values])
#                 f.write(f"{dataset} | {avg_reach:.2f} | {avg_time:.2f}\n")
#             f.write("\n")

# # Example usage
# input_file = "output.txt"   # Replace with your input file
# output_file = "ff.csv" # Replace with your desired output file
# process_data(input_file, output_file)

import csv
from collections import defaultdict
import numpy as np

def process_data_to_csv(input_file, output_csv):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    data = defaultdict(lambda: defaultdict(list))
    current_cut = None

    for line in lines:
        line = line.strip()
        if line.startswith("new cut:"):
            current_cut = int(line.split(":")[-1].strip())
        elif line:
            parts = line.split()
            if len(parts) >= 5:
                dataset = " ".join(parts[2:-2])  # Extract dataset name
                reach = float(parts[-2])         # Extract reach value
                time_taken = float(parts[-1])    # Extract time taken
                data[current_cut][dataset].append((reach, time_taken))

    # Write results to CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Cut Value", "Dataset", "Avg Reach", "Avg Time Taken"])
        
        for cut, datasets in data.items():
            for dataset, values in datasets.items():
                avg_reach = np.mean([v[0] for v in values])
                avg_time = np.mean([v[1] for v in values])
                writer.writerow([cut, dataset, f"{avg_reach:.2f}", f"{avg_time:.2f}"])

# Example usage
input_file = "output.txt"    # Replace with your input file
output_csv = "output.csv"  # Replace with your desired output CSV file
process_data_to_csv(input_file, output_csv)
