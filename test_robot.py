from utils.generate_response import Planner
from utils.sem_map import Semantic_Mapping
from utils.thor_env_code_v2 import ThorEnvCode
import prior
import pickle
from utils.procthor_config import Config as proc_Config
import json
import numpy as np
from collections import Counter, OrderedDict
import torch
import open_clip
from utils.sem_map import Semantic_Mapping
import cv2

def set_third_view(controller_env, third_camera_position_dict, third_camera_rotation_dict):
    robot_position_dict = controller_env.last_event.metadata["agent"]["position"]
    robot_rotation_dict = controller_env.last_event.metadata["agent"]["rotation"]
    # 根据y值进行处理
    robot_camera_y = robot_rotation_dict["y"]
    if abs(robot_camera_y - 270) < 5:
        third_camera_position_dict = dict(x=robot_position_dict["x"] + 1, y=2.3, z=robot_position_dict["z"] - 0.5)
        third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)
    elif abs(robot_camera_y - 180) < 5:
        third_camera_position_dict = dict(x=robot_position_dict["x"] - 0.5, y=2.3, z=robot_position_dict["z"] - 1)
        third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)
    elif abs(robot_camera_y - 90) < 5:
        third_camera_position_dict = dict(x=robot_position_dict["x"] - 1, y=2.3, z=robot_position_dict["z"] + 0.5)
        third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)
    elif abs(robot_camera_y - 0) < 5:
        third_camera_position_dict = dict(x=robot_position_dict["x"] - 1, y=2.3, z=robot_position_dict["z"] - 0.5)
        third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)
    else:
        pass

    controller_env.step(
        action="UpdateThirdPartyCamera",
        thirdPartyCameraId=0,
        position=third_camera_position_dict,
        rotation=third_camera_rotation_dict,
        fieldOfView=90
    )



kitchens_val = [f"FloorPlan{i}" for i in range(21, 26)]
living_rooms_val = [f"FloorPlan{200 + i}" for i in range(21, 26)]
bedrooms_val = [f"FloorPlan{300 + i}" for i in range(21, 26)]
bathrooms_val = [f"FloorPlan{400 + i}" for i in range(21, 26)]
dataset_val = kitchens_val + living_rooms_val + bathrooms_val + bedrooms_val

# 开始执行具体的环境配置
args = proc_Config
dataset = prior.load_dataset("procthor-10k")
dataset_train = dataset["train"]
controller = ThorEnvCode(args, use_CloudRendering=False)

# load clip for semantic feature map
open_clip_model = "ViT-H-14"
torch.autograd.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("Initializing model...")
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    open_clip_model, pretrained="./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"
)
clip_model.cuda("cuda:1")
clip_model.eval()

# 增加第三视角


# 读取信息
total_cat2idx_path = proc_Config.total_cat2idx_alfred_path
total_cat2idx = json.load(open(total_cat2idx_path))

sem_map = Semantic_Mapping(
        args=proc_Config,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        total_cat2idx=total_cat2idx
    )
sem_map.reset()

# 仅测试失败案例
with open("./log_file/llava_vln_parsed_response_v8_val_result_v2.pkl", "rb") as f:
    result_info_dict = pickle.load(f)
instruction_dict_list = json.load(open("./data/spaced_parse_instruction_easy_v8_val.json"))
new_instruction_dict_list = []
for val_dict_id, val_dict_one in enumerate(instruction_dict_list):
    result_info_dict_one = result_info_dict[val_dict_id]
    is_success = result_info_dict_one["success"]
    if not is_success:
        new_instruction_dict_list.append(val_dict_one)
# check_list = [7, 17, 42, 55, 91, 108, 115, 119]
check_list = [7]
# -------------------main loop--------------------------
for val_dict_id, val_dict_one in enumerate(new_instruction_dict_list):
    # if val_dict_id not in check_list:
    #     continue
    instruction = val_dict_one["instruction"].strip(" ").strip("\"")
    # house_id = int(val_dict_one["house_id"])
    house_id = 24
    rgb_frame, depth_frame = controller.load_scene(house_name=dataset_train[2333])
    success = False
    task_type = val_dict_one["task_type"]
    pddl_params = val_dict_one["pddl_params"]
    cleaned_objects = set()

    # 获取机器人位姿
    robot_position_dict = controller.last_event.metadata["agent"]["position"]
    robot_rotation_dict = controller.last_event.metadata["agent"]["rotation"]

    third_camera_position_dict = dict(x=robot_position_dict["x"] + 1, y=2.3, z=robot_position_dict["z"] - 0.5)
    third_camera_rotation_dict = dict(x=45, y=robot_rotation_dict["y"], z=0)

    event = controller.step(
        action="AddThirdPartyCamera",
        position=third_camera_position_dict,
        rotation=third_camera_rotation_dict,
        fieldOfView=90
    )

    while True:
        print("Enter command (w/a/s/d/q to quit):")
        # command = input().lower()
        command = input()
        # if keyboard.is_pressed('w'):
        if command == "w":
            controller.step("MoveAhead")
            set_third_view(controller, third_camera_position_dict, third_camera_rotation_dict)
        if command == "w_t":
            third_camera_position_dict["x"] += 0.5
            controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position=third_camera_position_dict,
                rotation=third_camera_rotation_dict,
                fieldOfView=90
            )
        if command == "s":
            # elif keyboard.is_pressed('s'):
            controller.step("MoveBack")
            set_third_view(controller, third_camera_position_dict, third_camera_rotation_dict)
        if command == "s_t":
            third_camera_position_dict["x"] -= 0.5
            controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position=third_camera_position_dict,
                rotation=third_camera_rotation_dict,
                fieldOfView=90
            )
        if command == "a":
            # elif keyboard.is_pressed('a'):
            controller.step("MoveLeft")
            set_third_view(controller, third_camera_position_dict, third_camera_rotation_dict)
        if command == "a_t":
            third_camera_position_dict["z"] -= 0.5
            controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position=third_camera_position_dict,
                rotation=third_camera_rotation_dict,
                fieldOfView=90
            )
        if command == "d":
            # elif keyboard.is_pressed('d'):
            controller.step("MoveRight")
            set_third_view(controller, third_camera_position_dict, third_camera_rotation_dict)
        if command == "d_t":
            third_camera_position_dict["z"] += 0.5
            controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position=third_camera_position_dict,
                rotation=third_camera_rotation_dict,
                fieldOfView=90
            )
        if command == "left":
            # elif keyboard.is_pressed('left'):
            controller.step("RotateLeft")
            set_third_view(controller, third_camera_position_dict, third_camera_rotation_dict)
        if command == "left_t":
            third_camera_rotation_dict["y"] += 15
            controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position=third_camera_position_dict,
                rotation=third_camera_rotation_dict,
                fieldOfView=90
            )
        if command == "right":
            # elif keyboard.is_pressed('right'):
            controller.step("RotateRight")
            set_third_view(controller, third_camera_position_dict, third_camera_rotation_dict)
        if command == "right_t":
            third_camera_rotation_dict["y"] -= 15
            controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position=third_camera_position_dict,
                rotation=third_camera_rotation_dict,
                fieldOfView=90
            )
        if command == "up":
            #  elif keyboard.is_pressed('up'):
            controller.step(
                action="LookUp",
                degrees=1
            )
        if command == "up_t":
            third_camera_rotation_dict["x"] -= 15
            controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position=third_camera_position_dict,
                rotation=third_camera_rotation_dict,
                fieldOfView=90
            )
        if command == "down":
            # elif keyboard.is_pressed('down'):
            controller.step(
                action="LookDown",
                degrees=1
            )
        if command == "down_t":
            third_camera_rotation_dict["x"] += 15
            controller.step(
                action="UpdateThirdPartyCamera",
                thirdPartyCameraId=0,
                position=third_camera_position_dict,
                rotation=third_camera_rotation_dict,
                fieldOfView=90
            )
        if command == "g_force":
            # elif keyboard.is_pressed('enter'):
            event = controller.step(
                action="PickupObject",
                objectId=next(
                    obj["objectId"] for obj in controller.last_event.metadata["objects"]
                    if obj["objectType"] == "SoapBottle"
                ),
                forceAction=True
            )
        if command == "g":
            # elif keyboard.is_pressed('enter'):
            event = controller.step(
                action="PickupObject",
                objectId=next(
                    obj["objectId"] for obj in controller.last_event.metadata["objects"]
                    if obj["objectType"] == "SoapBottle"
                ),
                forceAction=False
            )
            # print(controller.last_event)
        if command == "put":
            # elif keyboard.is_pressed('shift'):
            action = dict(action="PutObject",
                objectId=next(
                    obj["objectId"] for obj in controller.last_event.metadata["objects"]
                    if obj["objectType"] == "SinkBasin"
                ),
                forceAction=False,
                placeStationary=True)
            controller.step(action)
            # print(controller.last_event)
        if command == "put_force":
            # elif keyboard.is_pressed('shift'):
            action = dict(action="PutObject",
                objectId=next(
                    obj["objectId"] for obj in controller.last_event.metadata["objects"]
                    if obj["objectType"] == "SinkBasin"
                ),
                forceAction=True,
                placeStationary=True)
            controller.step(action)
        if command == "put_force_g":
            # elif keyboard.is_pressed('shift'):
            action = dict(action="PutObject",
                objectId=next(
                    obj["objectId"] for obj in controller.last_event.metadata["objects"]
                    if obj["objectId"].split("|")[0] == "GarbageCan"
                ),
                forceAction=True,
                placeStationary=True)
            controller.step(action)
            # print(controller.last_event)
        if command == "open_force":
            # elif keyboard.is_pressed('shift'):
            action = dict(action="OpenObject",
                objectId=next(
                    obj["objectId"] for obj in controller.last_event.metadata["objects"]
                    if obj["objectType"] == "Fridge"
                ),
                forceAction=True)
            controller.step(action)
            # print(controller.last_event)
        if command == "open":
            # elif keyboard.is_pressed('shift'):
            action = dict(action="OpenObject",
                objectId=next(
                    obj["objectId"] for obj in controller.last_event.metadata["objects"]
                    if obj["objectType"] == "Fridge"
                ),
                forceAction=False)
            controller.step(action)
            # print(controller.last_event)
        if command == "close_force":
            # elif keyboard.is_pressed('shift'):
            action = dict(action="CloseObject",
                objectId=next(
                    obj["objectId"] for obj in controller.last_event.metadata["objects"]
                    if obj["objectType"] == "Fridge"
                ),
                forceAction=True)
            controller.step(action)
            # print(controller.last_event)
        if command == "close":
            # elif keyboard.is_pressed('shift'):
            action = dict(action="CloseObject",
                objectId=next(
                    obj["objectId"] for obj in controller.last_event.metadata["objects"]
                    if obj["objectType"] == "Fridge"
                ),
                forceAction=False)
            controller.step(action)
            # print(controller.last_event)
        if command == "4":
            # elif keyboard.is_pressed('4'):
            controller.step(
                action="MoveHeldObjectLeft",
                forceVisible=True
            )
        if command == "6":
            # elif keyboard.is_pressed('6'):
            controller.step(
                action="MoveHeldObjectRight",
                forceVisible=True
            )
        if command == "8":
            # elif keyboard.is_pressed('8'):
            controller.step(
                action="MoveHeldObjectUp",
                forceVisible=True
            )
        if command == "2":
            # elif keyboard.is_pressed('2'):
            controller.step(
                action="MoveHeldObjectDown",
                forceVisible=True
            )
        if command == "1":
            # elif keyboard.is_pressed('1'):
            controller.step(
                action="MoveHeldObjectAhead",
                forceVisible=True
            )
        if command == "3":
            # elif keyboard.is_pressed('3'):
            controller.step(
                action="MoveHeldObjectBack",
                forceVisible=True
            )
        if command == "7":
            # elif keyboard.is_pressed('7'):
            object_id = controller.last_event.metadata['inventoryObjects'][0]["objectId"]
            print(object_id)
            controller.step(
                action="DisableObject",
                objectId=object_id
            )
            print(next(obj for obj in controller.last_event.metadata["objects"]
                       if obj["objectId"] == object_id))
        if command == "9":
            # elif keyboard.is_pressed('9'):
            object_id = controller.last_event.metadata['inventoryObjects'][0]["objectId"]
            controller.step(
                action="EnableObject",
                objectId=object_id
            )
            print(next(obj for obj in controller.last_event.metadata["objects"]
                       if obj["objectId"] == object_id))
        if command == "q":
            # elif keyboard.is_pressed('q'):
            print('Quit!')
            break

        # 读取第三视角相机
        third_camera_rgb = controller.last_event.third_party_camera_frames[0]
        print(type(third_camera_rgb))
        cv2.imwrite("0.png", third_camera_rgb)
        print(third_camera_position_dict)
        print(third_camera_rotation_dict)
        """
        # 检查object
        rgb, depth_frame, mask_list, info_dict = controller.get_obs()
        instance_segs = np.array(controller.last_event.instance_segmentation_frame)
        color_to_object_id = controller.last_event.color_to_object_id
        print("场景中有的物体:", color_to_object_id.values())
        print("-----------------------------------")
        print("预测物体", controller.frame_object_name_list)
        print("物体", command)

        # interact_mask = controller.seg.sem_seg_get_instance_mask_from_obj_type(command)
        interact_mask = controller.seg.get_instance_mask_from_obj_type(command)
        print(interact_mask)
        global_sem_map, global_sem_feature_map = sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
        nav_point_list = sem_map.get_sem_map_object_position_list(command)
        print(nav_point_list)
        # 保存一下mask
        # sem_vis = info_dict["vis_mask"]
        # sem_vis = cv2.resize(sem_vis, (500, 500))
        # cv2.imwrite("0.png", sem_vis)
        if interact_mask is not None:
            nz_rows, nz_cols = np.nonzero(interact_mask)
            instance_counter = Counter()
            for i in range(0, len(nz_rows), 1):
                x, y = nz_rows[i], nz_cols[i]
                instance = tuple(instance_segs[x, y])
                instance_counter[instance] += 1

            iou_scores = {}
            for color_id, intersection_count in instance_counter.most_common():
                union_count = np.sum(np.logical_or(np.all(instance_segs == color_id, axis=2), interact_mask.astype(bool)))
                iou_scores[color_id] = intersection_count / float(union_count)
            iou_sorted_instance_ids = list(OrderedDict(sorted(iou_scores.items(), key=lambda x: x[1], reverse=True)))

            # get the most common object ids ignoring the object-in-hand
            inv_obj = controller.last_event.metadata['inventoryObjects'][0]['objectId'] \
                if len(controller.last_event.metadata['inventoryObjects']) > 0 else None
            all_ids = [color_to_object_id[color_id] for color_id in iou_sorted_instance_ids
                       if color_id in color_to_object_id and color_to_object_id[color_id] != inv_obj]

            # print instance_ids
            instance_ids = [inst_id for inst_id in all_ids if inst_id is not None]
            print(instance_ids)
            instance_ids = controller.prune_by_any_interaction(instance_ids)
            print(instance_ids)

            # 位置坐标
            for instance_id in instance_ids:
                position = next(obj["position"] for obj in controller.last_event.metadata["objects"]
                                      if obj["objectId"] == instance_id)
                print("物体name: ", instance_id.split("|")[0])
                print(position)
                print("=============")

        else:
            print("无法匹配")
        """