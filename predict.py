import cv2
import os
import numpy as np
import torch
from torchvision import transforms
import time
from model.fcos import FCOSDetector
import random

# 配置
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
    score_threshold = 0.6
    nms_iou_threshold = 0.05
    max_detection_boxes_num = 5000

cfg = Config()

# 图像预处理
def preprocess_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transforms.ToTensor()(image)
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    return image

# 计算IoU
def compute_iou(fusion_box, boxes):
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

# Soft-NMS处理
def soft_nms(boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=2):
    N = boxes.shape[0]
    for i in range(N):
        maxpos = i + np.argmax(boxes[i:, 4])
        boxes[[i, maxpos]] = boxes[[maxpos, i]]
        for j in range(i + 1, N):
            iou = compute_iou(boxes[i], boxes[j].reshape(1, -1))
            iou = iou[0] if isinstance(iou, np.ndarray) else iou
            if method == 1:  # 线性
                if iou > Nt:
                    boxes[j, 4] = boxes[j, 4] * (1 - iou)
            elif method == 2:  # 高斯
                boxes[j, 4] = boxes[j, 4] * np.exp(-(iou ** 2) / sigma)
            elif method == 3:  # 原始NMS
                if iou > Nt:
                    boxes[j, 4] = 0
        boxes = boxes[boxes[:, 4] > threshold]
    return boxes[:, :4], boxes[:, 4]

# 加载模型
def load_model():
    model = FCOSDetector(config=Config).to(torch.device('cuda:0'))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load("./checkpoint/model_600.pth", map_location=torch.device('cpu')))
    model = model.eval()
    return model

# 处理图像
def process_image(image_path, model):
    origin_img_bgr = cv2.imread(image_path)
    origin_img_h, origin_img_w = origin_img_bgr.shape[:2]

    # 预处理
    resized_origin_img = cv2.resize(origin_img_bgr, (640, 640))
    origin_image = preprocess_img(resized_origin_img).to(device)

    # 模型推理
    start_t = time.time()
    with torch.no_grad():
        origin_out = model(origin_image.unsqueeze(dim=0))
    end_t = time.time()
    cost_t = 1000 * (end_t - start_t)
    print("Processing time: %.2f ms" % cost_t)

    origin_scores, origin_classes, origin_boxes = origin_out
    origin_boxes = origin_boxes[0].cpu().numpy()
    origin_classes = origin_classes[0].cpu().numpy().tolist()
    origin_scores = origin_scores[0].cpu().numpy().tolist()

    # 过滤掉低分数框
    all_origin_boxes = []
    for i, box in enumerate(origin_boxes):
        if origin_scores[i] < Config.score_threshold:
            continue
        adjusted_box = [int(box[0]), int(box[1]), int(box[2]), int(box[3]), origin_scores[i]]
        all_origin_boxes.append(adjusted_box)

    boxes = [x[:4] for x in all_origin_boxes]
    scores = [x[4] for x in all_origin_boxes]
    boxes_with_scores = np.hstack([np.array(boxes), np.array(scores).reshape(-1, 1)])

    # Soft-NMS
    boxes, scores = soft_nms(boxes_with_scores, sigma=0.5, Nt=0.3, threshold=0.001, method=2)
    print(boxes)

    # 恢复原始图像大小
    original_scale = (origin_img_w / 640, origin_img_h / 640)
    adjusted_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = int(x1 * original_scale[0])
        y1 = int(y1 * original_scale[1])
        x2 = int(x2 * original_scale[0])
        y2 = int(y2 * original_scale[1])
        adjusted_boxes.append([x1, y1, x2, y2])

    return origin_img_bgr, adjusted_boxes

def random_color():
    return [random.randint(0, 255) for _ in range(3)]

# 绘制图像与检测框
def draw_boxes(image, boxes, folder_name):
    result_img = image.copy()

    # 先将结果图像转换为BGRA格式，以便支持透明度
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2BGRA)

    # 创建一个文件夹保存裁剪的字符图像
    crop_folder = os.path.join('./output/crops', folder_name)
    os.makedirs(crop_folder, exist_ok=True)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box

        # 随机选择颜色
        color = (0, 130, 0)

        # 创建半透明背景：透明度设置为100（0~255之间的值）
        overlay = result_img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (*color, 100), -1)  # (color, 100) 表示透明度为100
        cv2.addWeighted(overlay, 0.4, result_img, 1 - 0.4, 0, result_img)

        # 绘制边框：使用不透明颜色绘制边框
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (*color, 255), 2)  # 边框使用完全不透明

        # 裁剪并保存字符图像
        crop_img = image[y1:y2, x1:x2]
        crop_path = os.path.join(crop_folder, f"{i}.png")
        cv2.imwrite(crop_path, crop_img)

    # 将图像转换回BGR格式，准备保存
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGRA2BGR)

    return result_img

if __name__ == "__main__":
    # 加载模型
    model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    origin_root = "./data/test/"
    names = os.listdir(origin_root)

    for name in names:
        # 处理图像
        image_path = os.path.join(origin_root, name)
        folder_name = os.path.splitext(name)[0]  # 使用文件名作为文件夹名称
        origin_img_bgr, boxes = process_image(image_path, model)

        # 绘制框并保存裁剪区域
        result_img = draw_boxes(origin_img_bgr, boxes, folder_name)

        # 保存结果图像
        save_path = os.path.join("./output", f"resu lt_{name}")
        cv2.imwrite(save_path, result_img)
