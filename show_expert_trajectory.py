import pickle
import os
import pdb
pdb.set_trace()

file_path = "/data1/wzy/ovmm_dataset/rl_agent_openvla/104348082_171512994_8067/obs_data.pkl"

with open(file_path, "rb") as f:
    data_dict = pickle.load(f)

print(type(data_dict))