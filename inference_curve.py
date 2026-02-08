import argparse
import cv2
import glob
import os

import numpy as np
from tqdm import tqdm
import torch
from yaml import load

from basicsr.utils import img2tensor, tensor2img, imwrite
from basicsr.archs.femasr_arch import FeMaSRNet
from basicsr.utils.download_util import load_file_from_url

import torch

_ = torch.manual_seed(123)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex')

# from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr

from comput_psnr_ssim import calculate_ssim as ssim_gray
from comput_psnr_ssim import calculate_psnr as psnr_gray
# from skimage.metrics import peak_signal_noise_ratio as psnr_gray
# def ssim_gray(imgA, imgB, gray_scale=True):
#     if gray_scale:
#         score, diff = ssim(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY), full=True,
#                            multichannel=False)
#     # multichannel: If True, treat the last dimension of the array as channels. Similarity calculations are done independently for each channel then averaged.
#     else:
#         score, diff = ssim(imgA, imgB, full=True, multichannel=True)
#     return score
#
#
# def psnr_gray(imgA, imgB, gray_scale=True):
#     if gray_scale:
#         psnr_val = psnr(cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY), cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY))
#         return psnr_val
#     else:
#         psnr_val = psnr(imgA, imgB)
#         return psnr_val


pretrain_model_url = {
    'x4': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX4_model_g.pth',
    'x2': 'https://github.com/chaofengc/FeMaSR/releases/download/v0.1-pretrain_models/FeMaSR_SRX2_model_g.pth',
}

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def equalize_hist_color(img):
    # 使用 cv2.split() 分割 BGR 图像
    channels = cv2.split(img)
    eq_channels = []
    # 将 cv2.equalizeHist() 函数应用于每个通道
    for ch in channels:
        eq_channels.append(cv2.equalizeHist(ch))
    # 使用 cv2.merge() 合并所有结果通道
    eq_image = cv2.merge(eq_channels)
    return eq_image

    # def get_residue_structure_mean(self, tensor, r_dim=1):
    #     max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    #     min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    #     res_channel = (max_channel[0] - min_channel[0])
    #     mean = torch.mean(tensor, dim=r_dim, keepdim=True)
    #
    #     device = mean.device
    #     res_channel = res_channel / torch.max(mean, torch.full(size=mean.size(), fill_value=0.000001).to(device))
    #     return res_channel

def get_residue_structure_mean(tensor, r_dim=1):
    max_channel = torch.max(tensor, dim=r_dim, keepdim=True)  # keepdim
    min_channel = torch.min(tensor, dim=r_dim, keepdim=True)
    res_channel = (max_channel[0] - min_channel[0])
    mean = torch.mean(tensor, dim=r_dim, keepdim=True)
    device = mean.device
    res_channel = res_channel / torch.max(mean, torch.full(size=mean.size(), fill_value=0.000001).to(device))
    return res_channel
import torch.nn.functional as F
def check_image_size(x,window_size=128):
    _, _, h, w = x.size()
    mod_pad_h = (window_size  - h % (window_size)) % (
                window_size )
    mod_pad_w = (window_size  - w % (window_size)) % (
                window_size)
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
    return x

def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def main():
    """Inference demo for FeMaSR
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str,
                        default='/data_8T1/wangcong/dataset/UHD/lowlight/testing_set/input',
                        help='Input image or folder')
    parser.add_argument('-g', '--gt', type=str,
                        default='/data_8T1/wangcong/dataset/UHD/lowlight/testing_set/gt',
                        help='groundtruth image')

    parser.add_argument('-w', '--weight', type=str,
                        default='./experiments/014_FeMaSR_LQ_stage/models/',
                        help='path for model weights')
    parser.add_argument('-o', '--output', type=str, default='results/curve', help='Output folder')
    parser.add_argument('-t', '--text', type=str, default='text/', help='text folder')
    parser.add_argument('-s', '--out_scale', type=int, default=1, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='', help='Suffix of the restored image')
    parser.add_argument('--max_size', type=int, default=600,
                        help='Max image size for whole image inference, otherwise use tiled_test')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    for k in range(12): # net_g_500000.pth
        k = 10000 + k * 10000
        weight  = args.weight + 'net_g_' + str(k) + '.pth'
        enhance_weight_path = weight
        print('weight_path', enhance_weight_path)
        # set up the model

        EnhanceNet = FeMaSRNet(unit_num=3,
                                  number_block=5,
                                  num_heads=8,
                                  match_factor=4,
                                  ffn_expansion_factor=4,
                                  scale_factor=8,
                                  bias=True,
                                  LayerNorm_type='WithBias',
                                  attention_discrimination=True,
                                  ffn_discrimination=True,
                                  SR_Guidance=True,
                                  ffn_restormer=False).to(device)
        EnhanceNet.load_state_dict(torch.load(enhance_weight_path)['params'], strict=False)
        EnhanceNet.eval()
        # print_network(EnhanceNet)
        os.makedirs(args.output, exist_ok=True)
        if os.path.isfile(args.input):
            paths = [args.input]
        else:
            paths = sorted(glob.glob(os.path.join(args.input, '*')))
        ssim_all = 0
        psnr_all = 0
        lpips_all = 0
        loss_all = 0
        loss_sr_all = 0
        num_img = 0
        pbar = tqdm(total=len(paths), unit='image')

        psnr_list = []
        ssim_list = []
        loss_list = []
        loss_sr_list = []
        lpips_list = []

        for idx, path in enumerate(paths):
            img_name = os.path.basename(path)
            pbar.set_description(f'Test {img_name}')

            gt_path = args.gt
            file_name = path.split('/')[-1]

            gt_img = cv2.imread(os.path.join(gt_path, file_name), cv2.IMREAD_UNCHANGED)

            # print(gt_img)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            img_tensor = img2tensor(img).to(device) / 255.
            img_tensor = img_tensor.unsqueeze(0)
            b, c, h, w = img_tensor.size()
            print('b, c, h, w = img_tensor.size()', img_tensor.size())
            img_tensor = check_image_size(img_tensor)

            gt_tensor = img2tensor(gt_img).to(device) / 255.
            gt_tensor = gt_tensor.unsqueeze(0)

            with torch.no_grad():
                output, SR_output = EnhanceNet.restoration_network(img_tensor)
            output = output
            # output = sr_model.test(img_tensor, rain = img_tensor-output)
            # else:
            #     output = sr_model.test_tile(img_tensor)
            # output_img = output['out_final']

            # [2, 1, 0]
            # output_first = tensor2img(output_first)
            output = output[:, :, :h, :w]
            SR_output = SR_output[:, :, :h, :w]
            from torch.nn import functional as F
            loss = F.l1_loss(output, gt_tensor, reduction='none').mean()
            loss_sr = F.l1_loss(SR_output, gt_tensor, reduction='none').mean()

            output_img = tensor2img(output)
            gray = True
            # ssim = ssim_gray(output_img, gt_img, gray_scale=gray)
            # psnr = psnr_gray(output_img, gt_img, gray_scale=gray)



            ssim = ssim_gray(output_img, gt_img)
            psnr = psnr_gray(output_img, gt_img)

            lpips_value = lpips(2 * torch.clip(img2tensor(output_img).unsqueeze(0) / 255.0, 0, 1) - 1,
                                2 * img2tensor(gt_img).unsqueeze(0) / 255.0 - 1)
            loss = loss.data.cpu().data.numpy()
            loss = float(loss)
            loss_sr = loss_sr.data.cpu().data.numpy()
            loss_sr = float(loss_sr)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            loss_list.append(loss)
            loss_sr_list.append(loss_sr)
            # lpips_list.append(lpips.data.cpu().data.numpy())

            ssim_all += ssim
            psnr_all += psnr
            loss_all += loss
            loss_sr_all += loss_sr
            lpips_all += lpips_value
            num_img += 1
            print('num_img', num_img)
            print('ssim', ssim)
            print('psnr', psnr)
            print('loss', loss)
            print('loss_sr', loss_sr)
            print('lpips_value', lpips_value)
            save_path = os.path.join(args.output, f'{img_name}')
            # save_path_first = os.path.join(args.output + 'first/', f'{img_name}')
            imwrite(output_img, save_path)

            pbar.update(1)
        pbar.close()
        import numpy as np
        print('avg_ssim:%f' % (ssim_all / num_img))
        print('avg_psnr:%f' % (psnr_all / num_img))
        print('avg_lpips:%f' % (lpips_all / num_img))
        print('avg_loss:%f' % (loss_all / num_img))
        print('avg_loss_sr:%f' % (loss_sr_all / num_img))
        # losses = {'L1loss2': loss}


        psnr_mean, psnr_std = np.mean(psnr_list), np.std(psnr_list)

        print('psnr_mean',psnr_mean, 'psnr_std',psnr_std)
        ssim_mean, ssim_std = np.mean(ssim_list), np.std(ssim_list)
        # lpips_mean, lpips_std = np.mean(lpips_list), np.std(lpips_list)
        loss_mean, loss_std = np.mean(loss_list), np.std(loss_list)
        loss_sr_mean, loss_sr_std = np.mean(loss_sr_list), np.std(loss_sr_list)

        psnr_dict = {'mean': psnr_mean,  'std': psnr_std}
        ssim_dict = {'mean': ssim_mean,  'std': ssim_std}
        loss_dict = {'mean': loss_mean, 'std': loss_std}
        loss_sr_dict = {'mean': loss_sr_mean, 'std': loss_sr_std}
        # lpips_dict = {'mean': lpips_mean,  'std':lpips_std}

        psnr_dict_stats = {'iter': k, **{f'{k}': v for k, v in psnr_dict.items()}}

        ssim_dict_stats = {'iter': k, **{f'{k}': v for k, v in ssim_dict.items()}}

        loss_dict_stats = {'iter': k, **{f'{k}': v for k, v in loss_dict.items()}}
        loss_sr_dict_stats = {'iter': k, **{f'{k}': v for k, v in loss_sr_dict.items()}}

        # lpips_dict_stats = {'iter': k, **{f'{v}': v for k, v in lpips_dict.items()}}

        import json
        ensure_dir(args.text)
        with open(os.path.join(args.text, "log_psnr.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(psnr_dict_stats) + "\n")

        with open(os.path.join(args.text, "log_ssim.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(ssim_dict_stats) + "\n")

        with open(os.path.join(args.text, "log_loss.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(loss_dict_stats) + "\n")


        with open(os.path.join(args.text, "log_loss_sr.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps(loss_sr_dict_stats) + "\n")

if __name__ == '__main__':
    main()
