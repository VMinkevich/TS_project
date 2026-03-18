import os
import json

def log_results(results_dict, file_name, model_name, output_path="results/tables/metrics.json"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    # data[dataset][model] = results
    if file_name not in data:
        data[file_name] = {}

    data[file_name][model_name] = results_dict

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)