import os
import torch
import torch.optim.lr_scheduler as lr_scheduler
import logging
import torch.utils.data
import model
from tqdm import tqdm
import time
import argparse
from dataloader import build_test_dataloader, build_train_dataloader
import yaml
from pathlib import Path
import glob
import re

root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
runs_dir = os.path.join(root, "runs")
res_dir = os.path.join(root, "res")

# Adopted from yolov7 project
def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def resize_keypoints(keypoints, box, input_size):
    '''
    Inputs : Coordinates of keypoints in a cropped box
    Outputs : Coordinates of keypoints resized to input_size (256 x 256)
    '''
    box = box.unsqueeze(1)
    return keypoints * (box[:,:,2:] - box[:,:,:2]) / input_size + box[:,:,:2]

def train(opt):
    # Make a directory to save weights
    Path(increment_path(Path(runs_dir) / opt.name, exist_ok=opt.exist_ok))(parents=True, exist_ok=True)

    with open(opt.config) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    train_dataloader, val_dataloader = build_train_dataloader()

    net = model.A2J_model(num_classes=cfg['kp num'])
    net = net.cuda()

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(
        net.parameters(), lr=cfg['lr'], weight_decay=cfg['decay']
    )
    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma)

    logging.basicConfig(
        format="%(asctime)s %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        filename=os.path.join(res_dir, "train", opt.name+".log"),
        level=logging.INFO,
    )
    logging.info("======================================================")

    for epoch in range(opt.nepoch):
        net = net.train()
        train_loss_sum = 0.0
        timer = time.time()

        # Training loop
        for i, (img, gt_keypoints, box) in enumerate(tqdm(train_dataloader)):
            

            img, gt_keypoints = img.cuda(), gt_keypoints.cuda()
            dt_keypoints = net(img)
            optimizer.zero_grad()
            output = loss(dt_keypoints, gt_keypoints)
            output.backward()
            optimizer.step()
            train_loss_sum += output.item()
            # printing loss info

        scheduler.step()

        # time taken
        timer = time.time() - timer
        timer = timer / len(train_dataloader.dataset)
        print("==> time to learn 1 sample = %f (ms)" % (timer * 1000))

        train_loss_sum /= len(train_dataloader.dataset)
        print(f"mean train_loss = {train_loss_sum}")

        # Validation
        net = net.eval()
        resized_keypoints = torch.FloatTensor()

        val_loss_sum = 0.0

        for i, (img, gt_keypoints, box) in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():
                img, gt_keypoints = img.cuda(), gt_keypoints.cuda()
                dt_keypoints = net(img)
                output = loss(dt_keypoints, gt_keypoints)
                resized_keypoints = torch.cat(
                    (
                        resized_keypoints,
                        resize_keypoints(dt_keypoints.detach().cpu(), box, cfg['input size']),
                    )
                )
                val_loss_sum += output.item()

        val_loss_sum /= len(val_dataloader.dataset)

        log = f"Epoch#{epoch}: train loss={train_loss_sum:.4f}, validation loss={val_loss_sum:.4f}, lr = {scheduler.get_last_lr()[0]:.6f}%.6f"
        print(log)
        logging.info(log)

        saveNamePrefix = f"{runs_dir}/epoch#{epoch}_lr_{cfg['lr']:.5f}_wetD_{cfg['decay']:.5f}_stepSize_{opt.step_size}_gamma_{opt.gamma}"
        torch.save(net.state_dict(), saveNamePrefix + ".pt")

def test():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=128, help="batch size")
    parser.add_argument('--nepoch', type=int, default=30, help="number of epochs")
    parser.add_argument('--step-size', type=int, default=10, help="step size of schedular")
    parser.add_argument('--gamma', type=float, default=0.1, help="gamma of schedular")
    parser.add_argument('--weights', type=str, help='model.pt path(s)')
    parser.add_argument('--config', type=str, default='src/cfg.yaml', help="path to configuration")
    parser.add_argument('--name', type=str, default='exp', help="name of experiment")
    opt = parser.parse_args()

    print(opt)

    if opt.weights:
        test(opt)
    else:
        train(opt)