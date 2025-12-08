from utils.prompt_utils import get_prompt_type_v5 as get_prompt_type
from utils.procthor_config import Config as proc_Config
import open_clip

from utils.generate_response import Planner
from utils.sem_map import Semantic_Mapping
from utils.thor_env_code_v2 import ThorEnvCode
import prior
import json

val_file = "./data/spaced_parse_instruction_easy_v8_val.json"
# 加载测试数据集

instruction_dict_list = json.load(open(val_file))
reslut_list = []

# 开始执行具体的环境配置
args = proc_Config
dataset = prior.load_dataset("procthor-10k")
dataset_train = dataset["train"]
controller_env = ThorEnvCode(args, use_CloudRendering=True)
val_dict_one = instruction_dict_list[0]
house_id = int(val_dict_one["house_id"])
rgb_frame, depth_frame = controller_env.load_scene(house_name=dataset_train[house_id])
object_metadata = controller_env.last_event.metadata["objects"]
for obj_ind in range(len(object_metadata)):
    obj_name = object_metadata[obj_ind]["objectId"].split("|")[0]
    # receptacle = object_metadata[obj_ind]["receptacle"]
    # obj_dict_ind = object_metadata[obj_ind]["objectId"]
    if "door" in obj_name.lower():
        print(object_metadata[obj_ind])

exit()