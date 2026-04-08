import ultralytics

from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="/home/mrajaraman/do-not-modify/image-to-coco-json-converter/output/train.json",
    save_dir="/home/mrajaraman/master-thesis-dragonfly/yolo-model",
    use_keypoints=True,
)