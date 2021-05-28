import sys, getopt
import json

### Help program to grab COCO classes for use

json_file = './annotations/instances_val2017.json'

with open(json_file,'r') as COCO:
    js = json.loads(COCO.read())
    categories = js["categories"]
    for category in categories:
        print(category["name"])