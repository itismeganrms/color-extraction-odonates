

# # This Python code converts a dataset in YOLO format into the COCO format. 
# # The YOLO dataset contains images of bottles and the bounding box annotations in the 
# # YOLO format. The COCO format is a widely used format for object detection datasets.

# # The input and output directories are specified in the code. The categories for 
# # the COCO dataset are also defined, with only one category for "bottle". A dictionary for the COCO dataset is initialized with empty values for "info", "licenses", "images", and "annotations".

# # The code then loops through each image in the input directory. The dimensions 
# # of the image are extracted and added to the COCO dataset as an "image" dictionary, 
# # including the file name and an ID. The bounding box annotations for each image are 
# # read from a text file with the same name as the image file, and the coordinates are 
# # converted to the COCO format. The annotations are added to the COCO dataset as an 
# # "annotation" dictionary, including an ID, image ID, category ID, bounding box coordinates,
# # area, and an "iscrowd" flag.

# # The COCO dataset is saved as a JSON file in the output directory.

# import json
# import os
# from PIL import Image

# # Set the paths for the input and output directories
# input_dir = '/home/mrajaraman/dataset/first_batch_pngs/data/images/train/'
# annotation_dir = '/home/mrajaraman/dataset/first_batch_pngs/data/labels/train'
# output_dir = '/home/mrajaraman/dataset/cocoo'

# # Define the categories for the COCO dataset
# categories = [{"id": 0, "name": "wings",
#                "id": 1, "name": "head",
#                "id": 2, "name": "torso",
#                "id": 3, "name": "tail"}]

# # Define the COCO dataset dictionary
# coco_dataset = {
#     "info": {},
#     "licenses": [],
#     "categories": categories,
#     "images": [],
#     "annotations": []
# }

# # Loop through the images in the input directory
# for id_num, image_file in enumerate(os.listdir(input_dir)):
    
#     # Load the image and get its dimensions
#     image_path = os.path.join(input_dir, image_file)
#     image = Image.open(image_path)
#     width, height = image.size
    
#     # Add the image to the COCO dataset
#     image_dict = {
#         "id": int(id_num),
#         "width": width,
#         "height": height,
#         "file_name": image_file
#     }
#     coco_dataset["images"].append(image_dict)
    
#     # Load the bounding box annotations for the image
#     with open(os.path.join(annotation_dir, f'{image_file.split(".")[0]}.txt')) as f:
#         annotations = f.readlines()
    
#     # Loop through the annotations and add them to the COCO dataset
#     for ann in annotations:
#         # x, y, w, h = map(float, ann.strip().split()[1:])
#         # x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
#         # x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
#         x, y, w, h = map(float, ann.strip().split()[1:])
#         x_min = (x - w/2) * width
#         y_min = (y - h/2) * height
#         x_max = (x + w/2) * width
#         y_max = (y + h/2) * height
        
#         # Polygon coordinates for rectangle
#         polygon = [
#             x_min, y_min,
#             x_max, y_min,
#             x_max, y_max,
#             x_min, y_max
#         ]
    
#         ann_dict = {
#             "id": len(coco_dataset["annotations"]),
#             "image_id": int(image_file.split('.')[0]),
#             "category_id": 0,
#             "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
#             "area": (x_max - x_min) * (y_max - y_min),
#             "iscrowd": 0
#         }
#         coco_dataset["annotations"].append(ann_dict)

# # Save the COCO dataset to a JSON file
# with open(os.path.join(output_dir, 'annotations.json'), 'w') as f:
#     json.dump(coco_dataset, f)


# import fiftyone as fo

# name = "dragonfly-train"
# dataset_dir1 = "/home/mrajaraman/dataset/first_batch_pngs/data/images/train"
# # dataset_dir2 = "/path/to/dir2/yolo-dataset"

# # Load only these specific classes
# classes = ["wings", "head", "torso", "tail"]

# # Create the dataset
# dataset = fo.Dataset.from_dir(
#     dataset_dir=dataset_dir1,
#     dataset_type=fo.types.YOLOv4Dataset,
#     label_field="labels",
#     name=name,
#     classes=classes,
# )

# # # Add additional directories of images
# # dataset.add_dir(
# #     dataset_path=dataset_dir2,
# #     dataset_type=fo.types.YOLOv4Dataset,
# #     label_field="ground_truth",
# #     classes=classes,
# # )



# # filenames = {}
# # samples_to_delete = []
# # for sample in dataset:
# #     if sample.filename not in filenames:
# #         filenames[sample.filename] = sample.id
# #     else:
# #         prev_sample_id = filenames[sample.filename]
# #         prev_sample = dataset[prev_sample_id]
# #         prev_sample.merge(sample, fields="ground_truth")
# #         samples_to_delete.append(sample.id)

# # dataset.delete_samples(samples_to_delete)


# export_dir = "/home/mrajaraman/dataset/cocoo/"
# label_field = "labels"  # for example

# # Export the subset
# dataset.export(
#     export_dir=export_dir,
#     dataset_type=fo.types.COCODetectionDataset,
#     label_field=label_field,
# )

import fiftyone as fo

dataset = fo.Dataset.from_dir(
    dataset_dir="/home/mrajaraman/dataset/first_batch_pngs/data/images/",
    dataset_type=fo.types.YOLOv4Dataset,
    label_field="ground_truth",
    classes=["wings", "head", "torso", "tail"]
)

dataset.export(
    export_dir="/home/mrajaraman/dataset/cocoo",
    dataset_type=fo.types.COCODetectionDataset,
    label_field="ground_truth"
)
