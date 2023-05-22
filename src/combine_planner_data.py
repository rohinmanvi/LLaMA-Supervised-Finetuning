import os
import json

input_directory = "data/"
output_file = "data/highway_planner_data_detailed_total.jsonl"

# Check if the output file exists and remove it
if os.path.exists(output_file):
    os.remove(output_file)

# Iterate through the JSONL files and combine them
i = 0
while True:
    input_file = f"{input_directory}highway_planner_data_detailed_{i}.jsonl"
    if not os.path.exists(input_file):
        break

    with open(input_file, "r") as infile, open(output_file, "a") as outfile:
        for line in infile:
            data = json.loads(line)
            outfile.write(json.dumps(data) + "\n")
    i += 1

print("All JSONL files have been combined into", output_file)
