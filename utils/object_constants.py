from collections import OrderedDict
import re

OBJECTS = [
    'AlarmClock',
    'Apple',
    'ArmChair',
    'BaseballBat',
    'BasketBall',
    'Bathtub',
    'BathtubBasin',
    'Bed',
    'Blinds',
    'Book',
    'Boots',
    'Bowl',
    'Box',
    'Bread',
    'ButterKnife',
    'Cabinet',
    'Candle',
    'Cart',
    'CD',
    'CellPhone',
    'Chair',
    'Cloth',
    'CoffeeMachine',
    'CounterTop',
    'CreditCard',
    'Cup',
    'Curtains',
    'Desk',
    'DeskLamp',
    'DishSponge',
    'Drawer',
    'Dresser',
    'Egg',
    'FloorLamp',
    'Footstool',
    'Fork',
    'Fridge',
    'GarbageCan',
    'Glassbottle',
    'HandTowel',
    'HandTowelHolder',
    'HousePlant',
    'Kettle',
    'KeyChain',
    'Knife',
    'Ladle',
    'Laptop',
    'LaundryHamper',
    'LaundryHamperLid',
    'Lettuce',
    'LightSwitch',
    'Microwave',
    'Mirror',
    'Mug',
    'Newspaper',
    'Ottoman',
    'Painting',
    'Pan',
    'PaperTowel',
    'PaperTowelRoll',
    'Pen',
    'Pencil',
    'PepperShaker',
    'Pillow',
    'Plate',
    'Plunger',
    'Poster',
    'Pot',
    'Potato',
    'RemoteControl',
    'Safe',
    'SaltShaker',
    'ScrubBrush',
    'Shelf',
    'ShowerDoor',
    'ShowerGlass',
    'Sink',
    'SinkBasin',
    'SoapBar',
    'SoapBottle',
    'Sofa',
    'Spatula',
    'Spoon',
    'SprayBottle',
    'Statue',
    'StoveBurner',
    'StoveKnob',
    'DiningTable',
    'CoffeeTable',
    'SideTable',
    'TeddyBear',
    'Television',
    'TennisRacket',
    'TissueBox',
    'Toaster',
    'Toilet',
    'ToiletPaper',
    'ToiletPaperHanger',
    'ToiletPaperRoll',
    'Tomato',
    'Towel',
    'TowelHolder',
    'TVStand',
    'Vase',
    'Watch',
    'WateringCan',
    'Window',
    'WineBottle',
]

OBJECTS_SINGULAR = [
    'alarmclock',
    'apple',
    'armchair',
    'baseballbat',
    'basketball',
    'bathtub',
    'bathtubbasin',
    'bed',
    'blinds',
    'book',
    'boots',
    'bowl',
    'box',
    'bread',
    'butterknife',
    'cabinet',
    'candle',
    'cart',
    'cd',
    'cellphone',
    'chair',
    'cloth',
    'coffeemachine',
    'countertop',
    'creditcard',
    'cup',
    'curtains',
    'desk',
    'desklamp',
    'dishsponge',
    'drawer',
    'dresser',
    'egg',
    'floorlamp',
    'footstool',
    'fork',
    'fridge',
    'garbagecan',
    'glassbottle',
    'handtowel',
    'handtowelholder',
    'houseplant',
    'kettle',
    'keychain',
    'knife',
    'ladle',
    'laptop',
    'laundryhamper',
    'laundryhamperlid',
    'lettuce',
    'lightswitch',
    'microwave',
    'mirror',
    'mug',
    'newspaper',
    'ottoman',
    'painting',
    'pan',
    'papertowel',
    'papertowelroll',
    'pen',
    'pencil',
    'peppershaker',
    'pillow',
    'plate',
    'plunger',
    'poster',
    'pot',
    'potato',
    'remotecontrol',
    'safe',
    'saltshaker',
    'scrubbrush',
    'shelf',
    'showerdoor',
    'showerglass',
    'sink',
    'sinkbasin',
    'soapbar',
    'soapbottle',
    'sofa',
    'spatula',
    'spoon',
    'spraybottle',
    'statue',
    'stoveburner',
    'stoveknob',
    'diningtable',
    'coffeetable',
    'sidetable'
    'teddybear',
    'television',
    'tennisracket',
    'tissuebox',
    'toaster',
    'toilet',
    'toiletpaper',
    'toiletpaperhanger',
    'toiletpaperroll',
    'tomato',
    'towel',
    'towelholder',
    'tvstand',
    'vase',
    'watch',
    'wateringcan',
    'window',
    'winebottle',
]

OBJECTS_LANG = ['alarm clock', 'apple', 'arm chair', 'baseball bat', 'basket ball', 'bathtub', 'bathtub basin', 'bed', 'blinds', 'book', 'boots', 'bowl', 'box', 'bread', 'butter knife', 'cabinet', 'candle', 'cart', 'CD', 'cell phone', 'chair', 'cloth', 'coffee machine', 'counter top', 'credit card', 'cup', 'curtains', 'desk', 'desk lamp', 'dish sponge', 'drawer', 'dresser', 'egg', 'floor lamp', 'footstool', 'fork', 'fridge', 'garbage can', 'glassbottle', 'hand towel', 'hand towel holder', 'house plant', 'kettle', 'key chain', 'knife', 'ladle', 'laptop', 'laundry hamper', 'laundry hamper lid', 'lettuce', 'light switch', 'microwave', 'mirror', 'mug', 'newspaper', 'ottoman', 'painting', 'pan', 'paper towel', 'paper towel roll', 'pen', 'pencil', 'pepper shaker', 'pillow', 'plate', 'plunger', 'poster', 'pot', 'potato', 'remote control', 'safe', 'salt shaker', 'scrub brush', 'shelf', 'shower door', 'shower glass', 'sink', 'sink basin', 'soap bar', 'soap bottle', 'sofa', 'spatula', 'spoon', 'spray bottle', 'statue', 'stove burner', 'stove knob', 'dining table', 'coffee table', 'side table', 'teddy bear', 'television', 'tennis racket', 'tissue box', 'toaster', 'toilet', 'toilet paper', 'toilet paper hanger', 'toilet paper roll', 'tomato', 'towel', 'towel holder', 'TV stand', 'vase', 'watch', 'watering can', 'window', 'wine bottle']


# def split_string_by_uppercase(input_string):
#     result = re.findall('[A-Z][^A-Z]*', input_string)
#     return result
#
# new_object_list = []
# for object_one in OBJECTS:
#     input_string = object_one
#     output = split_string_by_uppercase(input_string)
#     if len(output) > 1:
#         new_object_name = ""
#         for object_name_part_one in output:
#             new_object_name = new_object_name + object_name_part_one.lower() + " "
#         new_object_name = new_object_name.rstrip(" ")
#     else:
#         new_object_name = output[0].lower()
#     new_object_list.append(new_object_name)
#
# print(new_object_list)