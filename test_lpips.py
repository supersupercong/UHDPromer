import os
import torch
import cv2
import lpips
import numpy as np
device = torch.device('cuda')
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'


gt_path = '/data/wangliyan/dataset/UHD/UHD-Haze/test/gt'
results_path = "/data/wangliyan/code/UHDPromer-main/results/uhdpromer_UHDHaze"


lpips_fn = lpips.LPIPS(net='alex').to(device)   ###########LPIPS

imgsName = sorted(os.listdir(results_path))
gtsName = sorted(os.listdir(gt_path))

assert len(imgsName) == len(gtsName)

cumulative_lpips = 0

for i in range(len(imgsName)):
    print('Processing image: %s' % (imgsName[i]))
    res = cv2.imread(os.path.join(results_path, imgsName[i]), cv2.IMREAD_COLOR)
    gt = cv2.imread(os.path.join(gt_path, gtsName[i]), cv2.IMREAD_COLOR)

    gt_tensor = torch.tensor(np.array(gt)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    res_tensor = torch.tensor(np.array(res)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    lpips = lpips_fn(gt_tensor.to(device) * 2 - 1, res_tensor.to(device) * 2 - 1).squeeze().item()

    print('LPIPS is %.4f ' % (lpips))


    cumulative_lpips += lpips

print('Testing set, LPIPS is %.4f' % (cumulative_lpips / len(imgsName)))

print(results_path)

