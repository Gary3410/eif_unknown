import numpy as np

def check_task(self, task_type=None, task_label_dict=None):
    success = False
    if task_type is None:
        return success
    if task_type == "look_at_obj_in_light":
        ts = 2
        s = 0
        targets = self.get_targets(task_label_dict)
        toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
        inventory_objects = self.last_event.metadata['inventoryObjects']

        # check if the right object is in hand
        if len(inventory_objects) > 0 and inventory_objects[0]['objectId'] in [p['objectId'] for p in pickupables]:
            s += 1
        # check if the lamp is visible and turned on
        if np.any([t['isToggled'] and t['visible'] for t in toggleables]):
            s += 1

        if targets["sliced"]:
            ts += 1
            # check if some object was sliced
            objs_sliced = [t for t in pickupables if t["isSliced"]]
            if len(objs_sliced) > 0:
                s += 1

    elif task_type == "pick_and_place_simple":
        ts = 1
        s = 0

        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

        if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                           for r in receptacles if r['receptacleObjectIds'] is not None])
                   for p in pickupables]):
            s += 1

        if targets["sliced"]:
            ts += 1
            # check if some object was sliced
            objs_sliced = [t for t in pickupables if t["isSliced"]]
            if len(objs_sliced) > 0:
                s += 1

    elif task_type == "pick_and_place_with_movable_recep":
        ts = 3
        s = 0

        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
        movables = self.get_objects_with_name_and_prop(targets['mrecep'], 'pickupable')

        pickup_in_place = [p for p in pickupables for m in movables
                           if 'receptacleObjectIds' in p and m['receptacleObjectIds'] is not None
                           and p['objectId'] in m['receptacleObjectIds']]
        movable_in_place = [m for m in movables for r in receptacles
                            if 'receptacleObjectIds' in r and r['receptacleObjectIds'] is not None
                            and m['objectId'] in r['receptacleObjectIds']]
        # check if the object is in the final receptacle
        if len(pickup_in_place) > 0:
            s += 1
        # check if the movable receptacle is in the final receptacle
        if len(movable_in_place) > 0:
            s += 1
        # check if both the object and movable receptacle stack is in the final receptacle
        if np.any([np.any([p['objectId'] in m['receptacleObjectIds'] for p in pickupables]) and
                   np.any([r['objectId'] in m['parentReceptacles'] for r in receptacles]) for m in movables
                   if m['parentReceptacles'] is not None and m['receptacleObjectIds'] is not None]):
            s += 1

        if targets["sliced"]:
            ts += 1
            # check if some object was sliced
            objs_sliced = [t for t in pickupables if t["isSliced"]]
            if len(objs_sliced) > 0:
                s += 1

    elif task_type == "pick_clean_then_place_in_recep":
        ts = 3
        s = 0

        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

        objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                         if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
        objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was cleaned
        if len(objs_cleaned) > 0:
            s += 1
        # check if the object is both in the receptacle and clean
        if np.any([obj_id in objs_cleaned for obj_id in objs_in_place]):
            s += 1

        if targets["sliced"]:
            ts += 1
            # check if some object was sliced
            objs_sliced = [t for t in pickupables if t["isSliced"]]
            if len(objs_sliced) > 0:
                s += 1

    elif task_type == "pick_cool_then_place_in_recep":
        ts = 3
        s = 0

        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
        objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                         if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
        objs_cooled = [p['objectId'] for p in pickupables if p['objectId'] in self.cooled_objects]

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was cooled
        if len(objs_cooled) > 0:
            s += 1
        # check if the object is both in the receptacle and cold
        if np.any([obj_id in objs_cooled for obj_id in objs_in_place]):
            s += 1

        if targets["sliced"]:
            ts += 1
            # check if some object was sliced
            objs_sliced = [t for t in pickupables if t["isSliced"]]
            if len(objs_sliced) > 0:
                s += 1

    elif task_type == "pick_heat_then_place_in_recep":
        ts = 3
        s = 0

        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

        objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                         if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
        objs_heated = [p['objectId'] for p in pickupables if p['objectId'] in self.heated_objects]

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was heated
        if len(objs_heated) > 0:
            s += 1
        # check if the object is both in the receptacle and hot
        if np.any([obj_id in objs_heated for obj_id in objs_in_place]):
            s += 1

        if targets["sliced"]:
            ts += 1
            # check if some object was sliced
            objs_sliced = [t for t in pickupables if t["isSliced"]]
            if len(objs_sliced) > 0:
                s += 1

    elif task_type == "pick_two_obj_and_place":
        ts = 2
        s = 0

        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        if "," in targets["object"]:
            targets["object"] = targets["object"].split(",")
            pickupables_1 = self.get_objects_with_name_and_prop(targets['object'][0].strip(), 'pickupable')
            pickupables_2 = self.get_objects_with_name_and_prop(targets['object'][1].strip(), 'pickupable')

            if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                               for r in receptacles if r['receptacleObjectIds'] is not None])
                       for p in pickupables_1]):
                s += 1

            if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                               for r in receptacles if r['receptacleObjectIds'] is not None])
                       for p in pickupables_2]):
                s += 1

            if targets["sliced"]:
                ts += 2
                # check if some object was sliced
                objs_sliced = [t for t in pickupables_1 if t["isSliced"]]
                if len(objs_sliced) > 0:
                    s += 1
                objs_sliced = [t for t in pickupables_2 if t["isSliced"]]
                if len(objs_sliced) > 0:
                    s += 1
        else:
            pickupables = self.get_objects_with_name_and_prop(targets['object'].strip(), 'pickupable')
            s += min(np.max([sum([1 if r['receptacleObjectIds'] is not None
                                       and p['objectId'] in r['receptacleObjectIds'] else 0
                                  for p in pickupables])
                             for r in receptacles]), 2)
            if targets["sliced"]:
                ts += 2
                s += min(len([p for p in pickupables if p["isSliced"]]), 2)

    else:
        return success

    if ts == s:
        success = True

    response_dict = {"success": success,
                     "ts": ts,
                     "s": s}

    return response_dict


def check_complex_task(self, task_type=None, task_label_dict=None):
    success = False
    if task_type is None:
        return success
    if task_type == "get_a_drink_and_put_in_recep":
        ts = 3
        s = 0
        targets = self.get_targets(task_label_dict)

        # common beverages list
        # 需要变成procthor支持的物体
        common_beverages = ["coffee", "tea", "water", "juice", "soda", "milk", "smoothie", "cocktail", "wine"]
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

        objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                         if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
        # object_id.split("|")[0]
        objs_drinkable = [p['objectId'] for p in pickupables for d in common_beverages
                          if p['objectId'].lower() == d]

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1

        # check if some object is drinkable
        if len(objs_drinkable) > 0:
            s += 1

        # check if the object is both in the receptacle and drinkable
        if np.any([obj_id in objs_drinkable for obj_id in objs_in_place]):
            s += 1
    elif task_type == "pick_and_place_hard":
        ts = 3
        s = 0

        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        if "," in targets["object"]:
            targets["object"] = targets["object"].split(",")
            ts = len(targets["object"])
            for i in range(ts):
                pickupables = self.get_objects_with_name_and_prop(targets['object'][i].strip(), 'pickupable')
                if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                                for r in receptacles if r['receptacleObjectIds'] is not None])
                        for p in pickupables]):
                    s += 1
        else:
            pickupables = self.get_objects_with_name_and_prop(targets['object'].strip(), 'pickupable')
            s += min(np.max([sum([1 if r['receptacleObjectIds'] is not None
                                       and p['objectId'] in r['receptacleObjectIds'] else 0
                                  for p in pickupables])
                             for r in receptacles]), 3)
    elif task_type == "pick_slice_heat_then_place_in_recep":
        ts = 4
        s = 0

        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

        objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                         if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
        objs_heated = [p['objectId'] for p in pickupables if p['objectId'] in self.heated_objects]
        objs_sliced = []

        # check egg的属性是isBroken or isSliced
        for t in pickupables:
            if np.any(t['isSliced']):
                objs_sliced.append(t)

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was sliced
        if len(objs_sliced) > 0:
            s += 1
        # check if some object was heated
        if len(objs_heated) > 0:
            s += 1
        # check if the object is the same one
        if np.any([obj_id in objs_sliced and obj_id in objs_heated for obj_id in objs_in_place]):
            s += 1

    elif task_type == "box_obj_in_recep":
        ts = 3
        s = 0

        # 是否需要增加候选
        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        if "," in targets["object"]:
            targets["object"] = targets["object"].split(",")
            ts = len(targets["object"])
            for i in range(ts):
                pickupables = self.get_objects_with_name_and_prop(targets['object'][i].strip(), 'pickupable')
                if np.any([np.any([
                    ((p['objectId'] in r['receptacleObjectIds']) and (r['receptacleObjectIds'].lower() == 'box'))
                    for r in receptacles if r['receptacleObjectIds'] is not None])
                           for p in pickupables]):
                    s += 1
        else:
            pickupables = self.get_objects_with_name_and_prop(targets['object'].strip(), 'pickupable')
            s += min(np.max([sum([1 if r['receptacleObjectIds'] is not None
                                       and p['objectId'] in r['receptacleObjectIds'] and r[
                                           'receptacleObjectIds'].lower() == 'box' else 0
                                  for p in pickupables])
                             for r in receptacles]), 3)

    elif task_type == "pick_place_obj_in_recep_then_slice":
        ts = 3
        s = 0

        # 有多个物体
        # 增加候选
        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

        objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                         if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
        objs_sliced = []
        for t in pickupables:
            if np.any(t['isSliced']):
                objs_sliced.append(t)

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was sliced
        if len(objs_sliced) > 0:
            s += 1
        # check if the object is both in the receptacle and sliced
        if np.any([obj_id in objs_sliced for obj_id in objs_in_place]):
            s += 1

    elif task_type == 'pick_place_then_toggle_obj':
        ts = 2
        s = 0
        # 需要增加候选？
        targets = self.get_targets(task_label_dict)
        toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                           for r in receptacles if r['receptacleObjectIds'] is not None])
                   for p in pickupables]):
            s += 1
        if np.any([t['isToggled'] for t in toggleables]):
            s += 1

    elif task_type == 'pick_clean_heat_then_place_in_recep':
        ts = 5
        s = 0

        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

        objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                         if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]
        objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]
        # heated中没有增加ClothesDryer
        objs_heated = [p['objectId'] for p in pickupables if p['objectId'] in self.heated_objects]

        # check if object is in the receptacle
        if len(objs_in_place) > 0:
            s += 1
        # check if some object was cleaned
        if len(objs_cleaned) > 0:
            s += 1
        # check if some object was heated
        if len(objs_heated) > 0:
            s += 1
        # check if the object is both in the receptacle and clean
        if np.any([obj_id in objs_cleaned for obj_id in objs_in_place]):
            s += 1

        if np.any([obj_id in objs_heated for obj_id in objs_in_place]):
            s += 1

    elif task_type == 'toggle_obj_then_pick_place_obj':
        ts = 2
        s = 0
        # 要处理多个物体
        # 增加候选
        targets = self.get_targets(task_label_dict)
        toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                           for r in receptacles if r['receptacleObjectIds'] is not None])
                   for p in pickupables]):
            s += 1
        if np.any([t['isToggled'] for t in toggleables]):
            s += 1

    elif task_type == 'open_recep_then_pick_obj':
        ts = 1
        s = 0

        targets = self.get_targets(task_label_dict)
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')

        if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                           for r in receptacles if r['receptacleObjectIds'] is not None])
                   for p in pickupables]):
            s += 1
    elif task_type == 'pick_clean_obj_in_light':
        ts = 3
        s = 0
        targets = self.get_targets(task_label_dict)
        toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')

        objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]
        objs_in_place = [p['objectId'] for p in pickupables for r in receptacles
                         if r['receptacleObjectIds'] is not None and p['objectId'] in r['receptacleObjectIds']]

        # check if the lamp is visible and turned on
        if np.any([t['isToggled'] for t in toggleables]):
            s += 1
        if len(objs_in_place) > 0:
            s += 1
            # check if some object was cleaned
        if len(objs_cleaned) > 0:
            s += 1

    elif task_type == "pick_wash_obj_in_recep_with_movable_recep":
        ts = 5
        s = 0
        targets = self.get_targets(task_label_dict)

        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        movables = self.get_objects_with_name_and_prop(targets['mrecep'], 'pickupable')
        toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')
        mrecep_common_beverages = ["Bowl", "Plate"]
        object_common_beverages = ["Spoon", "Fork"]
        # 增加新的候选
        for object_common_one in object_common_beverages:
            pickupables_add = self.get_objects_with_name_and_prop(object_common_one, "pickupable")
            pickupables.extend(pickupables_add)
        for mrecep_common_one in mrecep_common_beverages:
            movables_add = self.get_objects_with_name_and_prop(mrecep_common_one, "pickupable")
            movables.extend(movables_add)

        objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]
        objs_cleaned.extend([m["objectId"] for m in movables if m["objectId"] in self.cleaned_objects])

        pickup_in_place = [p for p in pickupables for m in movables
                           if 'receptacleObjectIds' in p and m['receptacleObjectIds'] is not None
                           and p['objectId'] in m['receptacleObjectIds']]
        movable_in_place = [m for m in movables for r in receptacles
                            if 'receptacleObjectIds' in r and r['receptacleObjectIds'] is not None
                            and m['objectId'] in r['receptacleObjectIds']]

        objs_in_place = pickup_in_place + movable_in_place

        # check if the object is in the final receptacle
        if len(pickup_in_place) > 0:
            s += 1
        # check if the movable receptacle is in the final receptacle
        if len(movable_in_place) > 0:
            s += 1
        # check if both the object and movable receptacle stack is in the final receptacle
        if np.any([np.any([p['objectId'] in m['receptacleObjectIds'] for p in pickupables]) and
                   np.any([r['objectId'] in m['parentReceptacles'] for r in receptacles]) for m in movables
                   if m['parentReceptacles'] is not None and m['receptacleObjectIds'] is not None]):
            s += 1

        # check if the object is both in the receptacle and clean
        if np.any([obj_id in objs_cleaned for obj_id in objs_in_place]):
            s += 1

        if np.any([t['isToggled'] for t in toggleables]):
            s += 1

    elif task_type == "pick_clean_slice_obj_and_place":
        ts = 3
        s = 0
        targets = self.get_targets(task_label_dict)

        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        toggleables = self.get_objects_with_name_and_prop(targets['toggle'], 'toggleable')

        if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                           for r in receptacles if r['receptacleObjectIds'] is not None])
                   for p in pickupables]):
            s += 1

        objs_sliced = []
        for t in pickupables:
            if np.any(t['isSliced']):
                objs_sliced.append(t)

        # check if some object was sliced
        if len(objs_sliced) > 0:
            s += 1

        if np.any([t['isToggled'] for t in toggleables]):
            s += 1

    elif task_type == "pick_obj_then_open_recep_and_place":
        ts = 1
        s = 0
        targets = self.get_targets(task_label_dict)

        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')

        object_common_beverages = ["Spoon"]
        # 增加新的候选
        for object_common_one in object_common_beverages:
            pickupables_add = self.get_objects_with_name_and_prop(object_common_one, "pickupable")
            pickupables.extend(pickupables_add)

        if np.any([np.any([p['objectId'] in r['receptacleObjectIds']
                           for r in receptacles if r['receptacleObjectIds'] is not None])
                   for p in pickupables]):
            s += 1

    elif task_type == "pick_clean_and_cool_obj":
        ts = 1
        s = 0
        targets = self.get_targets(task_label_dict)

        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
        objs_cooled = [p['objectId'] for p in pickupables if p['objectId'] in self.cooled_objects]
        objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]

        cool_and_clean_object = [p["objectId"] for p in pickupables if p["objectId"] in objs_cleaned and\
                                 p["objectId"] in objs_cooled]

        if len(cool_and_clean_object) > 0:
            s += 1

    elif task_type == "pick_clean_heat_obj_with_movable_recep":
        ts = 4
        s = 0
        targets = self.get_targets(task_label_dict)
        pickupables = self.get_objects_with_name_and_prop(targets['object'], 'pickupable')
        receptacles = self.get_objects_with_name_and_prop(targets['parent'], 'receptacle')
        movables = self.get_objects_with_name_and_prop(targets['mrecep'], 'pickupable')
        mrecep_common_beverages = ["Plate"]
        object_common_beverages = ["Egg", "Tomato", "Potato", "Lettuce"]
        # 增加新的候选
        for object_common_one in object_common_beverages:
            pickupables_add = self.get_objects_with_name_and_prop(object_common_one, "pickupable")
            pickupables.extend(pickupables_add)
        for mrecep_common_one in mrecep_common_beverages:
            movables_add = self.get_objects_with_name_and_prop(mrecep_common_one, "pickupable")
            movables.extend(movables_add)

        objs_heated = [p['objectId'] for p in pickupables if p['objectId'] in self.heated_objects]
        objs_cleaned = [p['objectId'] for p in pickupables if p['objectId'] in self.cleaned_objects]

        pickup_in_place = [p for p in pickupables for m in movables
                           if 'receptacleObjectIds' in p and m['receptacleObjectIds'] is not None
                           and p['objectId'] in m['receptacleObjectIds']]
        movable_in_place = [m for m in movables for r in receptacles
                            if 'receptacleObjectIds' in r and r['receptacleObjectIds'] is not None
                            and m['objectId'] in r['receptacleObjectIds']]

        clean_and_heated_object = [p['objectId'] for p in pickupables if p['objectId'] in objs_cleaned and p["objectId"] in objs_heated]

        # check if the object is in the final receptacle
        if len(pickup_in_place) > 0:
            s += 1
        # check if the movable receptacle is in the final receptacle
        if len(movable_in_place) > 0:
            s += 1
        # check if both the object and movable receptacle stack is in the final receptacle
        if np.any([np.any([p['objectId'] in m['receptacleObjectIds'] for p in pickupables]) and
                   np.any([r['objectId'] in m['parentReceptacles'] for r in receptacles]) for m in movables
                   if m['parentReceptacles'] is not None and m['receptacleObjectIds'] is not None]):
            s += 1

        # check if the object is both in the clean and hot
        if len(clean_and_heated_object) > 0:
            s += 1