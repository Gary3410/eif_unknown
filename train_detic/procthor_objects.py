from detectron2.data.datasets.register_coco import register_coco_instances
import os

categories_v2_fix = [{'id': 1, 'name': 'wine bottle'}, {'id': 2, 'name': 'arm chair'}, {'id': 3, 'name': 'counter top'}, {'id': 4, 'name': 'fork'}, {'id': 5, 'name': 'table top decor'}, {'id': 6, 'name': 'garbage bag'}, {'id': 7, 'name': 'desktop'}, {'id': 8, 'name': 'plate'}, {'id': 9, 'name': 'spoon'}, {'id': 10, 'name': 'dog bed'}, {'id': 11, 'name': 'sofa'}, {'id': 12, 'name': 'plunger'}, {'id': 13, 'name': 'pen'}, {'id': 14, 'name': 'fridge'}, {'id': 15, 'name': 'toilet'}, {'id': 16, 'name': 'toilet paper'}, {'id': 17, 'name': 'watch'}, {'id': 18, 'name': 'toaster'}, {'id': 19, 'name': 'washing machine'}, {'id': 20, 'name': 'television'}, {'id': 21, 'name': 'apple'}, {'id': 22, 'name': 'desk lamp'}, {'id': 23, 'name': 'teddy bear'}, {'id': 24, 'name': 'tissue box'}, {'id': 25, 'name': 'tennis racket'}, {'id': 26, 'name': 'pillow'}, {'id': 27, 'name': 'soap bar'}, {'id': 28, 'name': 'egg'}, {'id': 29, 'name': 'baseball bat'}, {'id': 30, 'name': 'bowl'}, {'id': 31, 'name': 'lettuce'}, {'id': 32, 'name': 'laundry hamper'}, {'id': 33, 'name': 'faucet'}, {'id': 34, 'name': 'potato'}, {'id': 35, 'name': 'bread'}, {'id': 36, 'name': 'door'}, {'id': 37, 'name': 'bottle'}, {'id': 38, 'name': 'sink'}, {'id': 39, 'name': 'paper towel roll'}, {'id': 40, 'name': 'desk'}, {'id': 41, 'name': 'box'}, {'id': 42, 'name': 'clothes dryer'}, {'id': 43, 'name': 'cart'}, {'id': 44, 'name': 'vase'}, {'id': 45, 'name': 'key chain'}, {'id': 46, 'name': 'microwave'}, {'id': 47, 'name': 'pan'}, {'id': 48, 'name': 'garbage can'}, {'id': 49, 'name': 'ottoman'}, {'id': 50, 'name': 'coffee table'}, {'id': 51, 'name': 'stool'}, {'id': 52, 'name': 'kettle'}, {'id': 53, 'name': 'painting'}, {'id': 54, 'name': 'spray bottle'}, {'id': 55, 'name': 'TV stand'}, {'id': 56, 'name': 'CD'}, {'id': 57, 'name': 'spatula'}, {'id': 58, 'name': 'remote control'}, {'id': 59, 'name': 'tomato'}, {'id': 60, 'name': 'side table'}, {'id': 61, 'name': 'dish sponge'}, {'id': 62, 'name': 'coffee machine'}, {'id': 63, 'name': 'statue'}, {'id': 64, 'name': 'salt shaker'}, {'id': 65, 'name': 'cup'}, {'id': 66, 'name': 'ladle'}, {'id': 67, 'name': 'cell phone'}, {'id': 68, 'name': 'mug'}, {'id': 69, 'name': 'safe'}, {'id': 70, 'name': 'pencil'}, {'id': 71, 'name': 'pepper shaker'}, {'id': 72, 'name': 'laptop'}, {'id': 73, 'name': 'candle'}, {'id': 74, 'name': 'shelving unit'}, {'id': 75, 'name': 'dining table'}, {'id': 76, 'name': 'dumbbell'}, {'id': 77, 'name': 'chair'}, {'id': 78, 'name': 'knife'}, {'id': 79, 'name': 'vacuum cleaner'}, {'id': 80, 'name': 'bed'}, {'id': 81, 'name': 'credit card'}, {'id': 82, 'name': 'pot'}, {'id': 83, 'name': 'window'}, {'id': 84, 'name': 'butter knife'}, {'id': 85, 'name': 'dresser'}, {'id': 86, 'name': 'soap bottle'}, {'id': 87, 'name': 'cloth'}, {'id': 88, 'name': 'house plant'}, {'id': 89, 'name': 'alarm clock'}, {'id': 90, 'name': 'boots'}, {'id': 91, 'name': 'basket ball'}, {'id': 92, 'name': 'book'}, {'id': 93, 'name': 'newspaper'}, {'id': 94, 'name': 'sink basin'}]


def _get_builtin_metadata():
    id_to_name = {x['id']: x['name'] for x in categories_v2_fix}
    thing_dataset_id_to_contiguous_id = {
        x['id']: i for i, x in enumerate(
            sorted(categories_v2_fix, key=lambda x: x['id']))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}


_PREDEFINED_SPLITS_OBJECTS365 = {
    "object_procthor_train": ("vision_dataset/maskrcnn_v3/rgb", "vision_dataset/maskrcnn_v3/instances_train_procthor.json"),
    # 80,000 images, 1,240,587 annotations
    "object_procthor_val": ("vision_dataset/maskrcnn_v3/rgb", "vision_dataset/maskrcnn_v3/instances_val_procthor.json")
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_OBJECTS365.items():
    register_coco_instances(
        key,
        _get_builtin_metadata(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )