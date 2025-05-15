import os
import shutil
import xml.etree.ElementTree as ET
import cv2

def create_voc_xml(image_path, annotations, save_path):
    root = ET.Element("annotation")

    filename = ET.SubElement(root, "filename")
    filename.text = os.path.basename(image_path)

    size = ET.SubElement(root, "size")
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read image {image_path}")
        return
    h, w, _ = img.shape
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")
    width.text = str(w)
    height.text = str(h)

    for category, box in annotations:
        obj = ET.SubElement(root, "object")
        name = ET.SubElement(obj, "name")
        name.text = category

        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        ymin = ET.SubElement(bndbox, "ymin")
        xmax = ET.SubElement(bndbox, "xmax")
        ymax = ET.SubElement(bndbox, "ymax")

        xmin.text = str(int(box[0]))
        ymin.text = str(int(box[1]))
        xmax.text = str(int(box[2]))
        ymax.text = str(int(box[3]))

    tree = ET.ElementTree(root)
    tree.write(save_path, encoding="utf-8", xml_declaration=True)


def convert_to_voc(image_folder, annotation_folder, voc_image_folder, voc_annotation_folder):
    os.makedirs(voc_image_folder, exist_ok=True)
    os.makedirs(voc_annotation_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')]

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        annotation_path = os.path.join(annotation_folder, image_file.rsplit('.',1)[0] + ".txt")

        if not os.path.exists(annotation_path):
            print(f"Warning: Annotation file not found for {image_file}")
            continue

        annotations = []
        with open(annotation_path, "r", encoding='utf-8') as f:
            for line in f:
                data = line.strip().split(',')
                if len(data) < 5:
                    print(f"Warning: Skipping invalid line in {annotation_path}: {line.strip()}")
                    continue

                category = "char"
                try:
                    x1, y1, x2, y2 = map(float, data[1:5])
                except ValueError:
                    print(f"Warning: Invalid coordinate values in {annotation_path}: {line.strip()}")
                    continue

                xmin, ymin = min(x1, x2), min(y1, y2)
                xmax, ymax = max(x1, x2), max(y1, y2)
                annotations.append((category, [xmin, ymin, xmax, ymax]))

        xml_path = os.path.join(voc_annotation_folder, image_file.rsplit('.',1)[0] + ".xml")
        create_voc_xml(image_path, annotations, xml_path)

        shutil.copy(image_path, os.path.join(voc_image_folder, image_file))


if __name__ == "__main__":
    train_image_folder = "../data/nancho/train/train_images/"
    train_annotation_folder = "../data/nancho/train/train_annotations/"
    test_image_folder = "../data/nancho/test/test_images/"
    test_annotation_folder = "../data/nancho/test/test_annotations/"

    voc_train_image_folder = "voc_train_images"
    voc_train_annotation_folder = "voc_train_annotations"
    voc_test_image_folder = "voc_test_images"
    voc_test_annotation_folder = "voc_test_annotations"

    convert_to_voc(train_image_folder, train_annotation_folder, voc_train_image_folder, voc_train_annotation_folder)
    convert_to_voc(test_image_folder, test_annotation_folder, voc_test_image_folder, voc_test_annotation_folder)
