import diplib as dip
import os

def main():
    output_dir = "/home/mrajaraman/dataset/testing/png"
    os.makedirs(output_dir, exist_ok=True)
    input_dir = "/home/mrajaraman/dataset/testing/"

    for filename in os.listdir(input_dir):
        if filename.endswith('.jpg'):

            image_path = os.path.join(input_dir, filename)
            image = dip.ImageRead(image_path)

            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")

            dip.ImageWritePNG(image, output_path)
            print(f"Image saved to {output_path}")
            
if __name__ == "__main__":
    main()