#!/usr/bin/python
import sys, os, json, glob
import xml.etree.ElementTree as ET

INITIAL_BBOX_ID = 1
PREDEF_CLASSES = {'char': 1}  # Predefined category list

# Utility functions
def get(root, name):
    return root.findall(name)

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError(f'Cannot find {name} in {root.tag}.')
    if length > 0 and len(vars) != length:
        raise NotImplementedError(f'The size of {name} should be {length}, but is {len(vars)}.')
    return vars[0] if length == 1 else vars

def convert(xml_paths, out_json):
    json_dict = {
        'images': [],
        'type': 'instances',
        'categories': [],
        'annotations': []
    }
    categories = PREDEF_CLASSES
    bbox_id = INITIAL_BBOX_ID

    for image_id, xml_f in enumerate(xml_paths):
        sys.stdout.write(f'\r>> Converting image {image_id + 1}/{len(xml_paths)}')
        sys.stdout.flush()

        try:
            tree = ET.parse(xml_f)
        except ET.ParseError as e:
            print(f"\nSkipping invalid XML: {xml_f}, error: {e}")
            continue

        root = tree.getroot()
        try:
            filename = get_and_check(root, 'filename', 1).text
            size = get_and_check(root, 'size', 1)
            width = int(get_and_check(size, 'width', 1).text)
            height = int(get_and_check(size, 'height', 1).text)
        except NotImplementedError as e:
            print(f"\nMissing tags in {xml_f}, skipped. Error: {e}")
            continue

        image = {
            'file_name': filename,
            'height': height,
            'width': width,
            'id': image_id + 1
        }
        json_dict['images'].append(image)

        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            if category not in categories:
                print(f"\nNew category '{category}' found. Added.")
                categories[category] = max(categories.values()) + 1
            category_id = categories[category]

            bbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bbox, 'xmax', 1).text)
            ymax = int(get_and_check(bbox, 'ymax', 1).text)

            if xmax <= xmin or ymax <= ymin:
                print(f"\nInvalid bbox: {xmin, ymin, xmax, ymax} in {xml_f}, skipped.")
                continue

            o_width = xmax - xmin
            o_height = ymax - ymin
            ann = {
                'area': o_width * o_height,
                'iscrowd': 0,
                'image_id': image_id + 1,
                'bbox': [xmin, ymin, o_width, o_height],
                'category_id': category_id,
                'id': bbox_id,
                'ignore': 0,
                'segmentation': []
            }
            json_dict['annotations'].append(ann)
            bbox_id += 1

    # Add category definitions
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)

    if not json_dict['images']:
        raise ValueError("No images found in converted JSON.")
    if not json_dict['annotations']:
        print("Warning: No annotations found.")

    with open(out_json, 'w') as f:
        json.dump(json_dict, f, indent=4)


if __name__ == '__main__':
    xml_path = r'../data/nancho/train/train_annotations'
    xml_files = glob.glob(os.path.join(xml_path, '*.xml'))
    convert(xml_files, r'../data/nancho/train/train.json')
