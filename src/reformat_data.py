import json

with open('data/highway_planner_sequence_data_incremental_final_2.jsonl', 'r') as f_in:
    with open('data/highway_planner_data_incremental.jsonl', 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            text = data['text']
            
            # split into observation-action pairs
            pairs = text.split('Observation:')
            for pair in pairs[1:]:  # ignore the first split which is empty
                new_data = {"text": "Observation:" + pair}
                f_out.write(json.dumps(new_data) + '\n')
