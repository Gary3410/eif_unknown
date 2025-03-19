import numpy as np


class InteractiveScript(object):
    def __init__(self, agent, sem_map):
        self.agent = agent
        self.sem_map = sem_map
        self.distance = 0

    def reset_distance(self):
        self.distance = 0

    def get_nearest_recv_object(self):
        object_position_dict = self.sem_map.get_object_position_dict()
        receptacle_object_list = [obj["objectId"].split("|")[0] for obj in self.agent.last_event.metadata["objects"] if obj["receptacle"]]
        distance_list = []
        object_name_list = []
        position_list_all = []
        for receptacle_object_one in receptacle_object_list:
            if receptacle_object_one not in object_position_dict.keys():
                continue
            position_list = object_position_dict[receptacle_object_one]
            for position_one in position_list:
                _, path = self.agent.parse_nav_action(position_one)
                distance = (len(path) - 1) * 0.25
                distance_list.append(distance)
                object_name_list.append(receptacle_object_one)
                position_list_all.append(position_one)
        # 直接要最小的就行
        nearest_index = np.argmin(np.asarray(distance_list))
        target_position = position_list_all[nearest_index]
        recv_object_name = object_name_list[nearest_index]

        response_dict = {"target_position": target_position,
                         "recv_object_name": recv_object_name}
        return response_dict

    def get_surround_point(self, object_position):
        surround_point_list = []
        bias_mat = np.asarray([[0.5, 0],
                               [0, 0.5],
                               [-0.5, 0],
                               [0, -0.5]])
        object_position = np.asarray(object_position)
        for i in range(4):
            surround_point_list.append(object_position + bias_mat[i])

        response_dict = {"surround_point_list": surround_point_list}
        return response_dict

    def nav_surround_for_target(self, input_dict):
        is_find = False
        target_object_name = input_dict["target"]
        nearest_recv_object_dict = self.get_nearest_recv_object()
        target_position = nearest_recv_object_dict["target_position"]
        recv_object_name = nearest_recv_object_dict["recv_object_name"]
        # 获取附近巡航点
        surround_point_list_dict = self.get_surround_point(target_position)
        surround_point_list = surround_point_list_dict["surround_point_list"]
        # 移动并环视寻找
        for nav_point in surround_point_list:
            self.agent.perspective_camera_view()
            distance_one = self.agent.goto_location_nav(nav_point)
            self.distance = self.distance + distance_one
            is_find = self.agent.check_target_frame_alfred(target_object_name)
            if is_find:
                # 如果找到了, 启动一下观察
                rgb, depth_frame, mask_list, info_dict = self.agent.get_obs()
                _, _ = self.sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update")
                break
        return is_find

    def nav_surround(self):
        nearest_recv_object_dict = self.get_nearest_recv_object()
        recv_object_position = nearest_recv_object_dict["target_position"]
        surround_point_list_dict = self.get_surround_point(recv_object_position)
        surround_point_list = surround_point_list_dict["surround_point_list"]
        # 移动并环视寻找
        for nav_point in surround_point_list:
            self.agent.perspective_camera_view()
            distance_one = self.agent.goto_location_nav(nav_point)
            self.distance = self.distance + distance_one
            # 移动过去后, 增加环视
            for _ in range(4):
                rgb, depth_frame, mask_list, info_dict = self.agent.get_obs()
                _, _ = self.sem_map.forward(rgb, depth_frame, mask_list, info_dict, task="update", only_seg=True)
                # event = self.agent.step(action="RotateRight", forceAction=True)
                _, _, _, _ = self.agent.to_thor_api_exec(["RotateRight"])