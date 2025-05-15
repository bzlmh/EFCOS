import os
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from pycocotools.coco import COCO
from torchvision import transforms

import numpy as np
import random
from PIL import Image, ImageEnhance


class DataAugment(object):
    def __init__(self):
        super(DataAugment, self).__init__()

    def random_flip_horizon(self, img, boxes):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        xmin = w - boxes[:, 2]
        xmax = w - boxes[:, 0]
        boxes[:, 0] = xmin
        boxes[:, 2] = xmax
        return img, boxes

    def random_flip_vertical(self, img, boxes):
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        h = img.height
        ymin = h - boxes[:, 3]
        ymax = h - boxes[:, 1]
        boxes[:, 1] = ymin
        boxes[:, 3] = ymax
        return img, boxes

    def random_bright(self, img, u=32):

        alpha = random.uniform(-u, u) / 255
        img = ImageEnhance.Brightness(img).enhance(1 + alpha)
        return img

    def random_contrast(self, img, lower=0.5, upper=1.5):
        alpha = random.uniform(lower, upper)
        img = ImageEnhance.Contrast(img).enhance(alpha)
        return img

    def add_gasuss_noise(self, img, mean=0, std=0.1):
        img = np.array(img)
        noise = np.random.normal(mean, std, img.shape)
        img = img + noise
        img = np.clip(img, 0, 255).astype(np.uint8)
        return Image.fromarray(img)

    def add_salt_noise(self, img):
        img = np.array(img)
        salt = np.random.randint(0, 2, img.shape)
        img[salt == 1] = 255
        return Image.fromarray(img)

    def add_pepper_noise(self, img):
        img = np.array(img)
        pepper = np.random.randint(0, 2, img.shape)
        img[pepper == 0] = 0
        return Image.fromarray(img)

    def mixup(self, img1, img2, box1, box2, alpha=2):
        weight = random.betavariate(alpha, alpha)

        # Convert images to NumPy arrays
        img1 = np.array(img1).astype(np.float32)
        img2 = np.array(img2).astype(np.float32)

        # Perform mixup on the images (element-wise)
        miximg = weight * img1 + (1 - weight) * img2

        # Clip the pixel values to be within valid range [0, 255] and convert back to uint8
        miximg = np.clip(miximg, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        miximg = Image.fromarray(miximg)

        # Return mixed image and corresponding boxes based on weight
        if weight > 0.5:
            return miximg, box1
        else:
            return miximg, box2


    def random_crop(self, img, boxes, crop_area=0.8):
        w, h = img.size
        crop_width = int(w * crop_area)
        crop_height = int(h * crop_area)

        # Randomly pick a crop area
        left = random.randint(0, w - crop_width)
        upper = random.randint(0, h - crop_height)

        img = img.crop((left, upper, left + crop_width, upper + crop_height))

        # Adjust boxes accordingly
        boxes = self.adjust_boxes_after_crop(boxes, left, upper, crop_width, crop_height)
        return img, boxes

    def adjust_boxes_after_crop(self, boxes, left, upper, crop_width, crop_height):
        boxes = boxes.copy()
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= upper
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, crop_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, crop_height)
        return boxes


class COCODataset(Dataset):
    def __init__(self, imgs_path, anno_path, resize_size=[640,640], is_train=True,  data_augmentation=True, save_dir=None):
        super().__init__()
        self.coco = COCO(anno_path)
        self.imgs_path = imgs_path
        self.anno_path = anno_path
        self.resize_size = resize_size
        self.save_dir = save_dir
        self.is_train = is_train

        self.data_augmentation = DataAugment()  # 实例化 DataAugment 类

        self.ids = [img_id for img_id in self.coco.imgs.keys()]
        self.category2id = {v: i + 1 for i, v in enumerate(self.coco.getCatIds())}
        self.id2category = {v: k for k, v in self.category2id.items()}

        self.mean = [0.40789654, 0.44719302, 0.47026115]
        self.std = [0.28863828, 0.27408164, 0.27809835]

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.imgs_path, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        ann = self.coco.loadAnns(ann_ids)

        ann = [o for o in ann if o['iscrowd'] == 0]
        boxes = [o['bbox'] for o in ann]
        boxes = np.array(boxes, dtype=np.float32)
        boxes[..., 2:] = boxes[..., 2:] + boxes[..., :2]

        if self.is_train:
            available_augmentations = [
                self.data_augmentation.random_flip_horizon,
                self.data_augmentation.random_flip_vertical,
                self.data_augmentation.random_bright,
                self.data_augmentation.random_crop,
                self.data_augmentation.add_gasuss_noise,
                self.data_augmentation.add_salt_noise,
                self.data_augmentation.add_pepper_noise,
                self.data_augmentation.mixup
            ]

            # Randomly select 2 to 3 enhancement operations
            num_augmentations = random.randint(1, 4)  # 随机选择 1 到 4 个操作
            augmentations = random.sample(available_augmentations, num_augmentations)
            random.shuffle(augmentations)
            for aug in augmentations:
                if aug == self.data_augmentation.random_flip_horizon:
                    img, boxes = aug(img, boxes)
                elif aug == self.data_augmentation.random_flip_vertical:
                    img, boxes = aug(img, boxes)
                elif aug == self.data_augmentation.random_bright:
                    img = aug(img)
                elif aug == self.data_augmentation.random_crop:
                    img, boxes = aug(img, boxes)
                elif aug == self.data_augmentation.add_gasuss_noise:
                    img = aug(img)
                elif aug == self.data_augmentation.add_salt_noise:
                    img = aug(img)
                elif aug == self.data_augmentation.add_pepper_noise:
                    img = aug(img)
                elif aug == self.data_augmentation.mixup:
                    img, boxes = aug(img, img, boxes, boxes)


        img = np.array(img)
        img, boxes = self.preprocess_img_boxes(img, boxes, self.resize_size)

        classes = [o['category_id'] for o in ann]
        classes = [self.category2id[c] for c in classes]

        img = transforms.ToTensor()(img)
        boxes = torch.from_numpy(boxes)
        classes = torch.LongTensor(classes)

        if self.save_dir is not None:
            self.save_image_with_boxes(img, boxes, index)

        return img, boxes, classes

    def __len__(self):
        return len(self.ids)

    def save_image_with_boxes(self, img, boxes, index):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(img.permute(1, 2, 0).numpy())

        for box in boxes:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

        os.makedirs(self.save_dir, exist_ok=True)
        plt.savefig(os.path.join(self.save_dir, f'output_image_{index}.png'))
        plt.close(fig)

    def preprocess_img_boxes(self, image, boxes, input_ksize):
        min_side, max_side = input_ksize
        h, w, _ = image.shape

        smallest_side = min(w, h)
        largest_side = max(w, h)
        scale = min_side / smallest_side
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        pad_w = 32 - nw % 32
        pad_h = 32 - nh % 32

        image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
        image_paded[:nh, :nw, :] = image_resized

        if boxes is None:
            return image_paded
        else:
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale
            return image_paded, boxes


    def collate_fn(self, data):
        imgs_list, boxes_list, classes_list = zip(*data)
        batch_size = len(boxes_list)
        pad_imgs_list = []
        pad_boxes_list = []
        pad_classes_list = []

        h_list = [int(s.shape[1]) for s in imgs_list]
        w_list = [int(s.shape[2]) for s in imgs_list]
        max_h = np.array(h_list).max()
        max_w = np.array(w_list).max()
        for i in range(batch_size):
            img = imgs_list[i]
            pad_imgs_list.append(transforms.Normalize(self.mean, self.std, inplace=True)(torch.nn.functional.pad(
                img, (0, int(max_w - img.shape[2]), 0, int(max_h - img.shape[1])), value=0.0
            )))

        max_num = max([boxes.shape[0] for boxes in boxes_list])
        for i in range(batch_size):
            pad_boxes_list.append(
                torch.nn.functional.pad(boxes_list[i], (0, 0, 0, max_num - boxes_list[i].shape[0]), value=-1)
            )
            pad_classes_list.append(
                torch.nn.functional.pad(classes_list[i], (0, max_num - classes_list[i].shape[0]), value=-1)
            )

        batch_boxes = torch.stack(pad_boxes_list)
        batch_classes = torch.stack(pad_classes_list)
        batch_imgs = torch.stack(pad_imgs_list)

        return batch_imgs, batch_boxes, batch_classes



def visualize(dataset, index=None):
    if index is None:
        index = random.randint(0, len(dataset) - 1)

    img, boxes, classes = dataset[index]

    # Convert tensor to numpy array and transpose to (H, W, C)
    img = img.permute(1, 2, 0).numpy()

    if isinstance(boxes, torch.Tensor):
        boxes = boxes.numpy()
    if isinstance(classes, torch.Tensor):
        classes = classes.numpy()

    fig, ax = plt.subplots(1)
    ax.imshow(img)

    for box in boxes:
        if box[0] < 0:
            continue
        rect = patches.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

    plt.show()




if __name__ == "__main__":
    save_dir = "../vis"

    os.makedirs(save_dir, exist_ok=True)

    dataset = COCODataset(
        "../data/kzsj/train/train_images",
        "../data/kzsj/train/train.json",
        data_augmentation=True,
        save_dir=save_dir
    )
    visualize(dataset, 200)
