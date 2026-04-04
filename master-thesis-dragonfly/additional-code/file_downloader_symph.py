print("Entered")

import pandas as pd
import urllib.request
import requests
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns
from functools import reduce

from bs4 import BeautifulSoup
from PIL import Image

print("Entering code")

multimedia = pd.read_csv("/home/mrajaraman/master-thesis-dragonfly/geo_statistics/netherlands_symphetrum_striolatum_occurrences_full.csv")

print("File loaded")

headers = {"User-Agent": "Mozilla/5.0"}

main_images = "/home/mrajaraman/dataset/Sympetrum_striolatum_images"

print("Starting downloads")
for _, row in multimedia.iterrows():
    i = row['gbifID']
    url = row['references']
    replacement_url = row['identifier']

    if pd.isna(url):
        img_data = requests.get(replacement_url).content
        # print("Downloaded from replacement URL")
        with open('image_name.jpg', 'wb') as handler:
            handler.write(img_data)

        filename = os.path.join(main_images, f"img_{i}.jpg")
        if os.path.exists(filename):
            print(f"Downloaded from replacement URL. File {filename} already exists. Skipping download.")
            continue
        else:
            with open(filename, "wb") as f:
                f.write(img_data)
                print(f"Downloaded from replacement URL. Image downloaded successfully and filename is {filename}.")
    else:
        try:
            page = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(page.text, "html.parser")
            og_image_tag = soup.find("meta", property="og:image")

            if og_image_tag and og_image_tag.get("content"):
                og_image_url = og_image_tag["content"]
                # print("Original OG Image URL:", og_image_url)

                # Force HTTPS if HTTP
                if og_image_url.startswith("http://"):
                    og_image_url = og_image_url.replace("http://", "https://", 1)
                    # print("Corrected OG Image URL:", og_image_url)
                
                if "/small/" in og_image_url:
                    og_image_url = og_image_url.replace("/small/", "/large/")
                elif "_thumbnail" in og_image_url:
                    og_image_url = og_image_url.replace("_thumbnail", "_image")
                else:
                    og_image_url = og_image_url
            
                try:
                    # print("OG Image URL:", og_image_url)
                    img_response = requests.get(og_image_url, headers=headers, timeout=10)
                    img_response.raise_for_status()
                    filename = os.path.join(main_images, f"img_{i}.jpg")
                    if os.path.exists(filename):
                        print(f"File {filename} already exists. Skipping download.")
                        continue
                    else:
                        # print(f"Downloading image to {filename}...")
                        with open(filename, "wb") as f:
                            f.write(img_response.content)
                        print(f"Image downloaded successfully and filename is {filename}.")
                except requests.exceptions.RequestException as e:
                    print("Error downloading image:", e)
            else:
                print("og:image not found.")
            time.sleep(1)  # Be polite to servers
        except requests.exceptions.RequestException as e:
            print("Error fetching page:", e)