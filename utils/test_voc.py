import os
import cv2
import xml.etree.ElementTree as ET

def draw_bboxes_on_image(image_path, xml_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Failed to load image {image_path}")
        return

    # Parse XML annotation
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return
    root = tree.getroot()

    # Iterate over all objects in the annotation
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        # Draw bounding box (green, thickness=2)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Prepare output path and save image
    os.makedirs(output_path, exist_ok=True)
    output_image_path = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_image_path, img)
    print(f"Saved image with bboxes to {output_image_path}")

def process_folder(image_folder, xml_folder, output_folder):
    # List all JPEG images
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        xml_path = os.path.join(xml_folder, image_file.rsplit('.', 1)[0] + '.xml')

        if os.path.exists(xml_path):
            draw_bboxes_on_image(image_path, xml_path, output_folder)
        else:
            print(f"Warning: No XML found for {image_file}")

if __name__ == "__main__":
    image_folder = "../data/nancho/train_images"
    xml_folder = "../data/nancho/train_annotations"
    output_folder = "./output_images"

    process_folder(image_folder, xml_folder, output_folder)
