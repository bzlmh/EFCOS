from tqdm import tqdm
import xml.etree.ElementTree as ET
import cv2
import torch
from model.fcos import FCOSDetector
from torchvision import transforms
import os
import time
import numpy as np


def preprocess_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    return image


def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                             module.eps, module.momentum,
                                             module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name, convertSyncBNtoBN(child))
    del module
    return module_output


def compute_iou(fusion_box, boxes):
    if len(boxes) == 0:
        return np.array([])  # 返回空数组

    fusion_box_min = fusion_box[:2]
    fusion_box_max = fusion_box[2:4]
    boxes_min = boxes[..., :2]
    boxes_max = boxes[..., 2:4]
    fusion_wh = fusion_box_max - fusion_box_min
    boxes_wh = boxes_max - boxes_min

    if fusion_wh[0] <= 0 or fusion_wh[1] <= 0:
        return np.zeros(boxes.shape[0])  # 返回全零数组
    if np.any(boxes_wh <= 0):
        return np.zeros(boxes.shape[0])  # 返回全零数组

    fusion_area = fusion_wh[0] * fusion_wh[1]
    boxes_area = boxes_wh[..., 0] * boxes_wh[..., 1]

    inter_min = np.maximum(fusion_box_min, boxes_min)
    inter_max = np.minimum(fusion_box_max, boxes_max)
    inter_wh = np.maximum(0, inter_max - inter_min)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    ious = inter_area / (fusion_area + boxes_area - inter_area)
    return ious


def soft_nms(boxes, sigma=0.3, Nt=0.3, threshold=0.001, method=2):
    N = boxes.shape[0]
    for i in range(N):
        maxpos = i + np.argmax(boxes[i:, 4])
        boxes[[i, maxpos]] = boxes[[maxpos, i]]
        for j in range(i + 1, N):
            iou = compute_iou(boxes[i], boxes[j].reshape(1, -1))
            iou = iou[0] if isinstance(iou, np.ndarray) else iou  # 确保 iou 是标量
            if method == 1:  # linear
                if iou > Nt:
                    boxes[j, 4] = boxes[j, 4] * (1 - iou)
            elif method == 2:  # Gaussian
                boxes[j, 4] = boxes[j, 4] * np.exp(-(iou ** 2) / sigma)
            elif method == 3:  # original NMS
                if iou > Nt:
                    boxes[j, 4] = 0
        boxes = boxes[boxes[:, 4] > threshold]
    return boxes[:, :4], boxes[:, 4]  # 返回框和得分


class Config:
    pretrained = False
    freeze_stage_1 = False
    freeze_bn = False
    fpn_out_channels = 256
    use_p5 = True
    class_num = 1
    use_GN_head = True
    prior = 0.01
    add_centerness = True
    cnt_on_reg = True
    strides = [8, 16, 32, 64, 128]
    limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
    score_threshold = 0.5
    nms_iou_threshold = 0.3
    max_detection_boxes_num = 1500


cfg = Config()


def evaluate_detection(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.4):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0, 0, 0  # 如果没有预测框或真实框，跳过计算

    tp = 0
    fp = 0
    fn = len(gt_boxes)

    gt_boxes = np.array(gt_boxes)

    matched_gt_indices = set()

    for i, pred_box in enumerate(pred_boxes):
        ious = compute_iou(pred_box, gt_boxes)
        max_iou = np.max(ious)
        max_iou_index = np.argmax(ious)

        if max_iou >= iou_threshold:
            tp += 1
            matched_gt_indices.add(max_iou_index)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt_indices)

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return precision, recall, f1


def load_annotation_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    ground_truth_boxes = []
    ground_truth_classes = []

    class_mapping = {
        'char': 0,
        'other': 1,
    }

    for obj in root.findall('object'):
        category_name = obj.find('name').text
        category_id = class_mapping.get(category_name, -1)

        if category_id == -1:
            print(f"Warning: Category '{category_name}' not found in class_mapping.")
            continue

        bndbox = obj.find('bndbox')
        bbox = [
            float(bndbox.find('xmin').text),
            float(bndbox.find('ymin').text),
            float(bndbox.find('xmax').text),
            float(bndbox.find('ymax').text)
        ]
        ground_truth_boxes.append(bbox)
        ground_truth_classes.append(category_id)

    return ground_truth_boxes, ground_truth_classes


# ...（省略之前的代码）

if __name__ == "__main__":
    model = FCOSDetector(config=Config).to(torch.device('cuda:0'))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoint/model_1000.pth", map_location=torch.device('cuda')))
    model = model.eval()
    model = convertSyncBNtoBN(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("===>success loading model")

    origin_root = "./data/CHD/test/test_images/"
    xml_root = "./data/CHD/test/test_annotations/"
    names = os.listdir(origin_root)
    target_size = (800, 800)

    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_images = 0

    for name in tqdm(names, desc="Processing images"):
        try:
            origin_img_bgr = cv2.imread(os.path.join(origin_root, name))
            origin_img_h, origin_img_w = origin_img_bgr.shape[:2]

            xml_file = os.path.join(xml_root, name.replace('.png', '.xml'))
            if os.path.exists(xml_file):
                ground_truth_boxes, ground_truth_classes = load_annotation_from_xml(xml_file)
            else:
                print(f"Warning: No annotation found for {name}")
                continue
            resized_origin_img = cv2.resize(origin_img_bgr, target_size)
            origin_image = preprocess_img(resized_origin_img).to(device)

            start_t = time.time()
            with torch.no_grad():
                origin_out = model(origin_image.unsqueeze(dim=0))
            end_t = time.time()
            cost_t = 1000 * (end_t - start_t)
            print(f"===>success processing {name}, cost time {cost_t:.2f} ms")

            origin_scores, origin_classes, origin_boxes = origin_out
            origin_boxes = origin_boxes[0].cpu().numpy()
            origin_classes = origin_classes[0].cpu().numpy().tolist()
            origin_scores = origin_scores[0].cpu().numpy().tolist()

            all_origin_boxes = []
            for i, box in enumerate(origin_boxes):
                if origin_scores[i] < Config.score_threshold:
                    continue
                adjusted_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), origin_scores[i]]
                all_origin_boxes.append(adjusted_box)

            boxes = [x[:4] for x in all_origin_boxes]
            scores = [x[4] for x in all_origin_boxes]
            boxes_with_scores = np.hstack([np.array(boxes), np.array(scores).reshape(-1, 1)])

            boxes, scores = soft_nms(boxes_with_scores, sigma=0.5, Nt=0.3, threshold=0.001, method=2)

            original_scale = (origin_img_w / target_size[1], origin_img_h / target_size[0])
            adjusted_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                x1 = int(x1 * original_scale[0])
                y1 = int(y1 * original_scale[1])
                x2 = int(x2 * original_scale[0])
                y2 = int(y2 * original_scale[1])
                adjusted_boxes.append([x1, y1, x2, y2])

            adjusted_boxes = np.array(adjusted_boxes)

            if len(adjusted_boxes) > 0 and len(ground_truth_boxes) > 0:
                # 评估并累加精度、召回率和F1分数
                precision, recall, f1 = evaluate_detection(adjusted_boxes, scores, ground_truth_boxes)
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_images += 1
            else:
                print(f"Skipping image {name} due to empty predictions or ground truths.")
        except Exception as e:
            print(f"Error processing image {name}: {e}")

    # 计算平均精度、召回率和F1分数
    if total_images > 0:
        avg_precision = total_precision / total_images
        avg_recall = total_recall / total_images
        avg_f1 = total_f1 / total_images

        # 输出平均评估结果
        print("\nAverage evaluation results:")
        print(f"Average Precision: {avg_precision:.4f}")
        print(f"Average Recall: {avg_recall:.4f}")
        print(f"Average F1 Score: {avg_f1:.4f}")
    else:
        print("No images were processed for evaluation.")

