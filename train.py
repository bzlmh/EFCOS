from model.fcos import FCOSDetector
import torch
from dataset.COCO_dataset import COCODataset
import math, time
import os
import numpy as np
import random
import torch.backends.cudnn as cudnn
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=300, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=4, help="size of each image batch")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_gpu", type=str, default='0', help="gpu ids to use")
opt = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.n_gpu

# Set random seeds for reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

# Load training dataset
train_dataset = COCODataset("./data/nancho/train/train_images",
                            './data/nancho/train/train.json')

# Initialize model and move to GPU
model = FCOSDetector().cuda()
model = torch.nn.DataParallel(model)
model.train()

BATCH_SIZE = opt.batch_size
EPOCHS = opt.epochs

# Data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                           collate_fn=train_dataset.collate_fn,
                                           num_workers=opt.n_cpu, worker_init_fn=np.random.seed(0))

# Optimizer and learning rate scheduling
steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMUP_STEPS = 300
WARMUP_FACTOR = 1.0 / 3.0
GLOBAL_STEPS = 0

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params_count = sum([torch.numel(p) for p in model_parameters])
print("parameters", params_count)

LR_INIT = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=LR_INIT, momentum=0.9, weight_decay=0.001)
lr_schedule = [40000, 160000]

# Learning rate adjustment function
def lr_func(step):
    lr = LR_INIT
    if step < WARMUP_STEPS:
        alpha = float(step) / WARMUP_STEPS
        warmup_factor = WARMUP_FACTOR * (1.0 - alpha) + alpha
        lr = lr * warmup_factor
    else:
        for i in range(len(lr_schedule)):
            if step < lr_schedule[i]:
                break
            lr *= 0.1
    return float(lr)

start_epoch = 0
checkpoint_path = "./checkpoint/model_{}.pth"

# Load latest checkpoint if available
for i in range(1, EPOCHS + 1):
    checkpoint_file = checkpoint_path.format(i)
    if os.path.isfile(checkpoint_file):
        start_epoch = i
        model.load_state_dict(torch.load(checkpoint_file))
        print(f"Model loaded from {checkpoint_file}")
        break

# Training loop
for epoch in range(start_epoch, EPOCHS):
    for epoch_step, data in enumerate(train_loader):
        batch_imgs, batch_boxes, batch_classes = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()

        lr = lr_func(GLOBAL_STEPS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        start_time = time.time()

        optimizer.zero_grad()
        losses = model([batch_imgs, batch_boxes, batch_classes])
        loss = losses[-1]
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        end_time = time.time()
        cost_time = int((end_time - start_time) * 1000)

        print("global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f cost_time:%dms lr=%.4e total_loss:%.4f" % \
            (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch,
             losses[0].mean(), losses[1].mean(), losses[2].mean(),
             cost_time, lr, loss.mean()))

        GLOBAL_STEPS += 1

    # Save model every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), "./checkpoint/model_{}.pth".format(epoch + 1))
