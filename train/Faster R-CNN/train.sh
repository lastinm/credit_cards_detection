#!/bin/sh

python3 train.py --epoch 10 --train_image_dir "/home/lastinm/PROJECTS/CV/Roboflow creidt-cards Computer Vision Project COCO/train"\
 --val_image_dir "/home/lastinm/PROJECTS/CV/Roboflow creidt-cards Computer Vision Project COCO/valid"\
 --train_coco_json "/home/lastinm/PROJECTS/CV/Roboflow creidt-cards Computer Vision Project COCO/train/_annotations.coco.json"\
 --val_coco_json "/home/lastinm/PROJECTS/CV/Roboflow creidt-cards Computer Vision Project COCO/valid/_annotations.coco.json"\
 --batch_size 8