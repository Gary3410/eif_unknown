from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories_v2_fix = [{"id": 1, "name": "apple", "supercategory": "indoor"}, {"id":2,"name":"banana","supercategory":"indoor"},{"id":3,"name":"bottle","supercategory":"indoor"},{"id":4,"name":"bowl","supercategory":"indoor"},{"id":5,"name":"countertop","supercategory":"indoor"},{"id":6,"name":"dining table","supercategory":"indoor"},{"id":7,"name":"dish sponge","supercategory":"indoor"},{"id":8,"name":"fridge","supercategory":"indoor"},{"id":9,"name":"garbage can","supercategory":"indoor"},{"id":10,"name":"microwave","supercategory":"indoor"},{"id":11,"name":"mug","supercategory":"indoor"},{"id":12,"name":"pepper shaker","supercategory":"indoor"},{"id":13,"name":"potato","supercategory":"indoor"},{"id":14,"name":"sink","supercategory":"indoor"},{"id":15,"name":"tomato","supercategory":"indoor"}]

# HomeAssistant_v1
def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories_v2_fix}
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(
            sorted(categories_v2_fix, key=lambda x: x['id']))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

# /data1/wzy/vision_dataset/HomeAssistant_v1
# /home/wzy/workplace/auto_anno/Detic/datasets/houseassistant_v2
# /home/wzy/workplace/Detic/datasets/vision_dataset/HomeAssistant_v1/valid
_PREDEFINED_SPLITS_OBJECTS365 = {
    "realscene_v1_train": ("houseassistant_v2/train", "houseassistant_v2/train/_annotations.coco.json"),
    # 80,000 images, 1,240,587 annotations
    "realscene_v1_val": ("houseassistant_v2/valid", "houseassistant_v2/valid/_annotations.coco.json")
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJECTS365.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )