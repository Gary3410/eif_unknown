
class Target_Object(object):
    def __init__(self):
        self.candidate_position_list = []
        self.candidate_target_obj_id_list = []
        self.inter_object_list = []

    def get_target(self, target_name=None):
        is_check = False
        select_target_id = None
        if target_name is not None:
            for object_id_one in self.inter_object_list:
                if target_name in object_id_one:
                    is_check = True
                    select_target_id = object_id_one
                    break
        if is_check:
            select_index = self.candidate_target_obj_id_list.index(select_target_id)
            nav_position = self.candidate_position_list[select_index]
            return nav_position, select_target_id

        if len(self.candidate_position_list) > 0:
            nav_position = self.candidate_position_list[0]
            target_object_id = self.candidate_target_obj_id_list[0]
            return nav_position, target_object_id
        else:
            return None, None

    def reset(self, candidate_position_list, candidate_target_obj_id_list):
        self.candidate_position_list = candidate_position_list
        self.candidate_target_obj_id_list = candidate_target_obj_id_list

    def update_target(self):
        self.candidate_target_obj_id_list.pop(0)
        self.candidate_position_list.pop(0)

    def add_inter_object(self, target_object_id):
        self.inter_object_list.append(target_object_id)

    def reset_inter_object(self):
        self.inter_object_list = []

