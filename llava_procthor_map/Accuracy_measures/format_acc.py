import os
import json
import numpy as np

base_path = "llava_vln_parsed_response_v8_easy_val_result_lora.json"
acc_number = 0
json_file = json.load(open(base_path))
for json_one in json_file:
    gt = json_one["GT"]
    predict = json_one["output"]
    gt_answer = gt.split(".")[0].split(" ")[-1]
    predict_answer = predict.split(".")[0].split(" ")[-1]
    if gt_answer == predict_answer:
        acc_number = acc_number + 1
print("acc: ", acc_number / len(json_file))