import numpy as np
class DoneAction(object):
    def __init__(self, agent):
        self.agent = agent
        self.done_action_list = []
        self.done_low_level_action_list = []
        self.done_low_level_action_target_list = []
        self.interactive_object_list = []
        self.nav_object_position_list = []
        self.pickup_object_list = []
        self.distance_all = 0

        self.slice_object_name = []
        self.slice_agent_position = []

    def reset(self):
        self.done_action_list = []
        self.done_low_level_action_list = []
        self.done_low_level_action_target_list = []
        self.interactive_object_list = []
        self.nav_object_position_list = []
        self.pickup_object_list = []
        self.distance_all = 0

        self.slice_object_name = []
        self.slice_agent_position = []

    def reset_distance(self):
        self.distance_all = 0

    def add_done_action(self, action_planning):
        self.done_action_list.append(action_planning)

    def add_low_level_action(self, low_level_action, target):
        self.done_low_level_action_list.append(low_level_action)
        self.done_low_level_action_target_list.append(target)
        if "SliceObject" in low_level_action:
            self.slice_object_name.append(target)

    def get_done_action_str(self):
        done_list = "\n"
        for action_one in self.done_action_list:
            done_list = done_list + action_one + "\n"
        return done_list

    def get_done_low_action_str(self):
        done_list = "\n"
        for low_action_id, low_action_one in enumerate(self.done_low_level_action_list):
            done_list = low_action_one + ": " + self.done_low_level_action_target_list[low_action_id] + "\n"
        return done_list.rstrip("\n")

    def reflective_action(self):
        # 由于一些问题, 导致上一个动作执行错误
        # 移除最后一个动作
        if len(self.done_action_list) > 0:
            self.done_action_list.pop(-1)
        if len(self.done_low_level_action_list) > 0:
            self.done_low_level_action_list.pop(-1)
        if len(self.done_low_level_action_target_list) > 0:
            self.done_low_level_action_target_list.pop(-1)

    def add_interactive_object(self, input_dict):
        # 记录交互物体的ID或name
        # Agent的位置, 视角等详细信息
        fov = self.agent.last_event.metadata["fov"]
        cameraHorizon = self.agent.last_event.metadata["agent"]["cameraHorizon"]
        robot_position = np.asarray(list(self.agent.last_event.metadata["agent"]["position"].values()))
        rotation = self.agent.last_event.metadata["agent"]["rotation"]['y']
        # 解析交互info_dict
        object_name = input_dict["target_name"]
        action = input_dict["low_action"]
        object_id = input_dict["target_object_id"]
        # 保存信息
        save_info_dict = {"fov": fov,
                          "cameraHorizon": cameraHorizon,
                          "robot_position": robot_position,
                          "rotation": rotation,
                          "object_name": object_name,
                          "action": action,
                          "object_id": object_id}
        if "pickup" in action.lower():
            self.pickup_object_list.append(object_name)
        self.interactive_object_list.append(save_info_dict)

    def add_nav_object_position(self, input_dict):
        # 记录巡航物体
        # Agent的位置, 视角等详细信息
        fov = self.agent.last_event.metadata["fov"]
        cameraHorizon = self.agent.last_event.metadata["agent"]["cameraHorizon"]
        robot_position = np.asarray(list(self.agent.last_event.metadata["agent"]["position"].values()))
        rotation = self.agent.last_event.metadata["agent"]["rotation"]['y']
        # 解析交互info_dict
        object_name = input_dict["target_name"]
        action = input_dict["low_action"]
        # 保存信息
        save_info_dict = {"fov": fov,
                          "cameraHorizon": cameraHorizon,
                          "robot_position": robot_position,
                          "rotation": rotation,
                          "object_name": object_name,
                          "action": action}
        self.nav_object_position_list.append(save_info_dict)

    def check_previous_pose(self, input_dict):
        response_dict = {}
        object_name = input_dict["target_name"]
        action = input_dict["low_action"]
        action_type = input_dict["action_type"]
        if action_type == "nav":
            for nav_action_dict_one in reversed(self.nav_object_position_list):
                object_name_one = nav_action_dict_one["object_name"]
                action_one = nav_action_dict_one["action"]
                if object_name_one in self.pickup_object_list:
                    continue
                if object_name == object_name_one and action_one == action:
                    self.set_previous_pose(nav_action_dict_one)
                    response_dict["success"] = True
                    response_dict["distance"] = self.distance_all
                    response_dict["object_name"] = object_name
                    response_dict["object_id"] = None
        elif action_type == "inter":
            for inter_action_dict_one in reversed(self.interactive_object_list):
                object_name_one = inter_action_dict_one["object_name"]
                action_one = inter_action_dict_one["action"]
                object_id = inter_action_dict_one["object_id"]
                if object_name == object_name_one and action_one == action:
                    self.set_previous_pose(inter_action_dict_one)
                    response_dict["success"] = True
                    response_dict["distance"] = self.distance_all
                    response_dict["object_name"] = object_name
                    response_dict["object_id"] = object_id
        else:
            pass
        if len(response_dict.keys()) < 3:
            response_dict["success"] = False
            response_dict["distance"] = 0
            response_dict["object_name"] = None
            response_dict["object_id"] = None
        return response_dict

    def set_previous_pose(self, input_dict):
        cameraHorizon = input_dict["cameraHorizon"]
        robot_position = input_dict["robot_position"]
        rotation = input_dict["rotation"]
        # 首先移动到历史位置
        distance_one = self.agent.goto_location_nav((robot_position[0], robot_position[-1]))
        # 调整视角
        self.agent.perspective_camera_view(cameraHorizon)
        # 调整转向
        self.agent.perspective_robot_angle(rotation)
        self.distance_all = self.distance_all + distance_one

    def get_sliced_object_position(self, target):
        assert len(self.slice_agent_position) == len(self.slice_object_name)
        if target in self.slice_object_name:
            select_index = self.slice_object_name.index(target)
            return self.slice_agent_position[select_index]
        else:
            return [-1, -1]





