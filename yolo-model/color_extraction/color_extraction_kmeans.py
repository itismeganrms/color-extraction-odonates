import pandas as pd
import cv2
import numpy as np

from ultralytics import YOLO
from matplotlib import pyplot as plt
from PIL import Image

from cv2 import THRESH_BINARY
from cv2 import imread, imwrite, resize, cvtColor, threshold
from sklearn.cluster import KMeans

from collections import Counter

class ColorExtraction:
    clusters = None
    images = None
    colors = None
    labels = None

    def __init__(self, clusters, image):
        self.clusters = clusters
        self.image = image

    def dominantColors(self):
        img_reshaped = self.image.reshape((-1,3))
        self.image = img_reshaped
        kmeans = KMeans(n_clusters = self.clusters)
        kmeans.fit(self.image)
        self.colors = kmeans.cluster_centers_
        self.labels = kmeans.labels_
        return self.colors.astype(int)
    
    def plot_colors(self, hist,colors):
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()] 
        # chart = np.zeros((50, 500, 3), np.uint8)
        chart = np.ones((50, 500, 3), np.uint8) * 255
        start = 0
        for i in range(self.clusters):
            end = start + hist[i] * 500
            r = int(colors[i][0])
            g = int(colors[i][1])
            b = int(colors[i][2])
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r, g, b), -1)
            start = end	
        return chart
    
    def plotHistogram(self):
        numLabels = np.arange(0, self.clusters+1)
        hist, _ = np.histogram(color_extraction.labels, bins=numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()
        colors = self.colors
        chart = ColorExtraction.plot_colors(self, hist,colors)
        return chart
    
    def plotfinalimage(images, titles, image_id):
        rows = int(np.ceil(len(images) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(6, rows * 3))
        axes = axes.flatten()

        for i, (img, title) in enumerate(zip(images, titles)):
            axes[i].imshow(img)
            axes[i].set_title(title,fontsize=12)
            axes[i].axis("off")

        for ax in axes[len(images):]:
            ax.axis('off')

        plt.tight_layout()
        # fig.suptitle("Extraction of colors and palettes using K-Means Clustering",fontsize=16)
        plt.savefig(f"/home/mrajaraman/master-thesis-dragonfly/results-and-images/yolo/color_palette/kmeans_symph_palette/{image_id}_color_palette.png")
        plt.show()
        print("Saved Final Image")

if __name__ == "__main__":
    # onnx_model = YOLO("/home/mrajaraman/master-thesis-dragonfly/yolo-model/runs/segment/train60/weights/best.onnx")
    best_model = YOLO("/home/mrajaraman/master-thesis-dragonfly/yolo-model/runs/segment/train64/weights/best.onnx")    
    best_model.names.update({0: 'head', 1: 'abdomen', 2: 'thorax', 3: 'wings'})

    # test_images = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/test_batch.csv")
    test_images = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/symph.csv")

    for image in test_images['Path']:
        print("Evaluating image" + image)
        try:
            Image.open(image).verify()
        except Exception:
            print(f"Corrupted image file: {image}")
            continue

        images = []
        titles = []
        test_image = image
        clusters = 5
        image_id = test_image.split("/")[-1].split(".")[0]

        # results = best_model(test_image, save=True)
        results = best_model(source=str(image), classes=[0,1,2], save=True)
        r = results[0]
        boxes = r.boxes
        classes = boxes.cls.cpu().tolist()
        masks = r.masks

        test_image = imread(test_image)
        img = cvtColor(test_image, cv2.COLOR_BGR2RGB)
        heading = "Original Image"
        images = [img]
        titles.append(heading)

        yolo_result = r.plot()
        yolo_results = cvtColor(yolo_result,cv2.COLOR_BGR2RGB)
        heading = "Prediction from trained model"
        images.append(yolo_results)
        titles.append(heading)

        class_names = r.names
        # print("Class names are ", class_names)
        class_ids = r.boxes.cls.cpu().numpy().astype(int)
        counter = Counter(class_ids)
        output = [(class_names[k], v) for k, v in counter.items()]
        output = dict(output)

        if masks is None:
            print("No detections / no masks found for image ID ", image_id)
        
        elif (output.get("head",0)==1 and output.get("thorax",0)==1 and output.get("abdomen",0)==1):
            for class_id, mask_tensor in enumerate(masks.data):
                class_name = class_names[class_id]
                part_mask = mask_tensor.cpu().numpy()
                part_resized = cv2.resize(part_mask, (test_image.shape[1], test_image.shape[0]))
                mask_uint = (part_resized * 255).astype(np.uint8)
                _, mask_binary = cv2.threshold(mask_uint, 127, 255, cv2.THRESH_BINARY)
                resultant_part = cv2.bitwise_and(img, img, mask=mask_binary)
                heading = f"Segmentation of {class_name}"
                images.append(resultant_part)
                titles.append(heading)

                mask_colored = np.any(resultant_part != 0, axis=2)
                colored_pixels = resultant_part[mask_colored]
                color_extraction = ColorExtraction(clusters, colored_pixels)
                colors = color_extraction.dominantColors()
                chart = color_extraction.plotHistogram()
                heading = f"Palette of {class_name}"
                images.append(chart)
                titles.append(heading)

            ColorExtraction.plotfinalimage(images, titles, image_id)
        else:
            print("Invalid count of objects found for image ID ", image_id)

