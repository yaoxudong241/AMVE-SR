import os
from tqdm import tqdm
import cv2
import torch
import numpy as np
import utils
import model.EDSR
import argparse
import torch.nn.functional as F
import torch.nn as nn
def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--net_class', type=str, default='SR',
                        help='Implemented model,such as vgg16, vgg16_bn')

    parser.add_argument('--model', type=str, default='vgg16_bn',
                        help='Implemented model,such as vgg16, vgg16_bn')
    parser.add_argument('--base_cam', type=str, default="grad_cam",
                        help='base cam, grad_cam or score_cam')
    parser.add_argument('--denoising', type=bool, default=False,
                        help='Whether to use SR-CAM with denoising or not')

    parser.add_argument('--target_class', type=int, default=None,
                        help='target_class, default model\'s predicted class')

    parser.add_argument("--checkpoint", type=str, default=r'weights\epoch_1000.pth',
                        help='checkpoint folder to use')
    parser.add_argument("--scale", type=int, default=int(scale_str),
                        help="super-resolution scale")
    parser.add_argument("--rgb_range", type=int, default=1,
                        help="maxium value of RGB")
    parser.add_argument("--n_colors", type=int, default=3,
                        help="number of color channels to use")

    parser.add_argument('--cuda', action='store_true', default=True,
                        help='use cuda')
    parser.add_argument("--upscale_factor", type=int, default=int(scale_str),
                        help='upscaling factor')
    parser.add_argument("--is_y", action='store_true', default=True,
                        help='evaluate on y channel, if False evaluate on RGB channels')

    parser.add_argument('--n_resblocks', type=int, default=16,
                        help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=64,
                        help='number of feature maps')
    parser.add_argument('--res_scale', type=float, default=1,
                        help='residual scaling')

    parser.add_argument('--subimage_x', type=int, default=206,
                        help='subimage_x')

    parser.add_argument('--subimage_y', type=int, default=206,
                        help='subimage_y')

    parser.add_argument('--subimage_size', type=int, default=100,
                        help='subimage_size')

    parser.add_argument('--layer_name', type=str, default="tail.1",
                        help='subimage_size')

    parser.add_argument('--target_layer', type=int, default=2,
                        help='target_layer, -1 represents the input layer')
    return parser


if __name__ == '__main__':

    scale_str = "8"
    seed_str = "176"

    subimg_x = 206
    subimg_y = 206
    subimg_size = 100

    val_path = r"val.txt"


    hrPath = r"data\HR"
    lrPath = r"data\LR\X{}".format(scale_str)
    paths = [r"data\Hirescam", r"data\layercam", r"data\SR_cam", r"data\lam"]

    opts = get_argparser().parse_args()

    model = model.EDSR.EDSR(opts)
    model.load_state_dict(torch.load(opts.checkpoint), strict=True)  # True)
    model.eval()

    cuda = opts.cuda
    device = torch.device('cuda' if cuda else 'cpu')
    if cuda:
        model = model.to(device)
    l1_criterion = nn.L1Loss()
    for root, dirs, files in os.walk(hrPath):
        for file in tqdm(files):
            if file.endswith('.png'):
                imname = hrPath +"//" + file

                Hirescam = paths[0] + "//" + file[0:-4] + "_head0_vis.png"
                layercam = paths[1] + "//" + file[0:-4] + "_head0_vis.png"
                SR_cam = paths[2] + "//" + file[0:-4] + "_head0_vis.png"
                lam = paths[3] + "//" + file[0:-4] + "_saliency_image.png"

                mythreshold = 0.15

                im_Hirescam = cv2.resize(cv2.imread(Hirescam),(64,64))[:,:,1]
                im_Hirescam[im_Hirescam>(mythreshold*255)] = 255
                im_Hirescam[im_Hirescam < (mythreshold*255)] = 0

                im_layercam = cv2.resize(cv2.imread(layercam),(64,64))[:,:,1]
                im_layercam[im_layercam>(mythreshold*255)] = 255
                im_layercam[im_layercam < (mythreshold*255)] = 0

                im_SR_cam = cv2.resize(cv2.imread(SR_cam),(64,64))[:,:,1]
                im_SR_cam[im_SR_cam>(mythreshold*255)] = 255
                im_SR_cam[im_SR_cam < (mythreshold*255)] = 0

                im_lam = cv2.imread(lam)[:,:,1]
                im_lam[im_lam>(mythreshold*255)] = 255
                im_lam[im_lam < (mythreshold*255)] = 0

                im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
                im_gt2 = im_gt
                im_gt = utils.modcrop(im_gt, opts.upscale_factor)

                im_l = cv2.imread(lrPath +"//" + file,
                                  cv2.IMREAD_COLOR)[:,
                       :, [2, 1, 0]]  # BGR to RGB

                Hirescam_crop_l = im_l * (np.expand_dims(im_Hirescam, axis=-1)/255)

                layercam_crop_l = im_l * (np.expand_dims(im_layercam, axis=-1) / 255)
                SRcam_crop_l = im_l * (np.expand_dims(im_SR_cam, axis=-1) / 255)
                lam_crop_l = im_l * (np.expand_dims(im_lam, axis=-1) / 255)

                im_l2 = im_l
                if len(im_gt.shape) < 3:
                    im_gt = im_gt[..., np.newaxis]
                    im_gt = np.concatenate([im_gt] * 3, 2)

                    im_l = im_l[..., np.newaxis]
                    im_l = np.concatenate([im_l] * 3, 2)

                    Hirescam_crop_l = Hirescam_crop_l[..., np.newaxis]
                    Hirescam_crop_l = np.concatenate([Hirescam_crop_l] * 3, 2)

                    layercam_crop_l = layercam_crop_l[..., np.newaxis]
                    layercam_crop_l = np.concatenate([layercam_crop_l] * 3, 2)

                    SRcam_crop_l = SRcam_crop_l[..., np.newaxis]
                    SRcam_crop_l = np.concatenate([SRcam_crop_l] * 3, 2)

                    lam_crop_l = lam_crop_l[..., np.newaxis]
                    lam_crop_l = np.concatenate([lam_crop_l] * 3, 2)

                im_input = im_l / 255.0
                im_input = np.transpose(im_input, (2, 0, 1))
                im_input = im_input[np.newaxis, ...]
                im_input = torch.from_numpy(im_input).float()

                Hirescam_crop_l_im_input = Hirescam_crop_l / 255.0
                Hirescam_crop_l_im_input = np.transpose(Hirescam_crop_l_im_input, (2, 0, 1))
                Hirescam_crop_l_im_input = Hirescam_crop_l_im_input[np.newaxis, ...]
                Hirescam_crop_l_im_input = torch.from_numpy(Hirescam_crop_l_im_input).float()

                layercam_crop_l_im_input = layercam_crop_l / 255.0
                layercam_crop_l_im_input = np.transpose(layercam_crop_l_im_input, (2, 0, 1))
                layercam_crop_l_im_input = layercam_crop_l_im_input[np.newaxis, ...]
                layercam_crop_l_im_input = torch.from_numpy(layercam_crop_l_im_input).float()

                SRcam_crop_l_im_input = SRcam_crop_l / 255.0
                SRcam_crop_l_im_input = np.transpose(SRcam_crop_l_im_input, (2, 0, 1))
                SRcam_crop_l_im_input = SRcam_crop_l_im_input[np.newaxis, ...]
                SRcam_crop_l_im_input = torch.from_numpy(SRcam_crop_l_im_input).float()

                lam_crop_l_im_input = lam_crop_l / 255.0
                lam_crop_l_im_input = np.transpose(lam_crop_l_im_input, (2, 0, 1))
                lam_crop_l_im_input = lam_crop_l_im_input[np.newaxis, ...]
                lam_crop_l_im_input = torch.from_numpy(lam_crop_l_im_input).float()

                im_gt = im_gt / 255.0
                im_gt = np.transpose(im_gt, (2, 0, 1))
                im_gt = im_gt[np.newaxis, ...]
                im_gt = torch.from_numpy(im_gt).float()

                if cuda:
                    im_input = im_input.to(device)
                    Hirescam_crop_l_im_input = Hirescam_crop_l_im_input.to(device)
                    layercam_crop_l_im_input = layercam_crop_l_im_input.to(device)
                    SRcam_crop_l_im_input = SRcam_crop_l_im_input.to(device)
                    lam_crop_l_im_input = lam_crop_l_im_input.to(device)

                    im_gt = im_gt.to(device)

                SR_out = model(im_input)
                Hirescam_SR_out = model(Hirescam_crop_l_im_input)
                layercam_SR_out = model(layercam_crop_l_im_input)
                SRcam_SR_out = model(SRcam_crop_l_im_input)
                lam_SR_out = model(lam_crop_l_im_input)

                # 边缘损失
                SR_patch = SR_out[:, :, subimg_x: subimg_x + subimg_size, subimg_y: subimg_y + subimg_size]
                Hirescam_SR_patch = Hirescam_SR_out[:, :, subimg_x: subimg_x + subimg_size, subimg_y: subimg_y + subimg_size]
                layercam_SR_patch = layercam_SR_out[:, :, subimg_x: subimg_x + subimg_size, subimg_y: subimg_y + subimg_size]
                SRcam_SR_patch = SRcam_SR_out[:, :, subimg_x: subimg_x + subimg_size, subimg_y: subimg_y + subimg_size]
                lam_SR_patch = lam_SR_out[:, :, subimg_x: subimg_x + subimg_size, subimg_y: subimg_y + subimg_size]

                HR_patch = im_gt[:, :, subimg_x: subimg_x + subimg_size, subimg_y: subimg_y + subimg_size]

                sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0)
                sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(
                    0).unsqueeze(0)

                # 扩展滤波器以匹配输入通道数
                sobel_x = sobel_x.repeat(3, 1, 1, 1).to("cuda:0")
                sobel_y = sobel_y.repeat(3, 1, 1, 1).to("cuda:0")

                labels_padded = F.pad(HR_patch, (1, 1, 1, 1), mode='replicate')
                grad_x_labels = F.conv2d(labels_padded, sobel_x, groups=3)
                grad_y_labels = F.conv2d(labels_padded, sobel_y, groups=3)

                edge_labels = torch.sqrt(grad_x_labels ** 2 + grad_y_labels ** 2)

                loss = l1_criterion(edge_labels * SR_patch,
                                    edge_labels * HR_patch)

                Hirescam_loss = l1_criterion(edge_labels * Hirescam_SR_patch,
                                    edge_labels * HR_patch)

                layercam_loss = l1_criterion(edge_labels * layercam_SR_patch,
                                    edge_labels * HR_patch)

                SRcam_loss = l1_criterion(edge_labels * SRcam_SR_patch,
                                    edge_labels * HR_patch)

                lam_loss = l1_criterion(edge_labels * lam_SR_patch,
                                    edge_labels * HR_patch)

                with open(val_path, 'a') as filetxt:
                    filetxt.write("===> Valid. filename:{}, edge_loss: {:.4f}, Hirescam: {:.4f}, layercam: {:.4f},  SRcam: {:.4f}, lam: {:.4f}\n".format(file,loss.item(),
                                                                               Hirescam_loss.item(),layercam_loss.item(),SRcam_loss.item(),lam_loss.item() ))


                # with open(filename_path, 'a') as filetxt:
                #     filetxt.write("===> Valid. name: {}\n".format(file))

                # print(targetPath+"//"+file)
                # SR_out_img = utils.tensor2np(SR_out.detach()[0])
                # cv2.imwrite(targetPath+"//"+file, SR_out_img[:, :, [2, 1, 0]])
                #
                # Hirescam_SR_out_img = utils.tensor2np(Hirescam_SR_out.detach()[0])
                # cv2.imwrite(targetPath+"//"+file[0:-4]+"_Hirescam.png", Hirescam_SR_out_img[:, :, [2, 1, 0]])
                #
                # layercam_SR_out_img = utils.tensor2np(layercam_SR_out.detach()[0])
                # cv2.imwrite(targetPath+"//"+file[0:-4]+"_layercam.png", layercam_SR_out_img[:, :, [2, 1, 0]])
                #
                # SRcam_SR_out_img = utils.tensor2np(SRcam_SR_out.detach()[0])
                # cv2.imwrite(targetPath + "//" + file[0:-4]+"_SRcam.png", SRcam_SR_out_img[:, :, [2, 1, 0]])
                #
                # lam_SR_out_img = utils.tensor2np(lam_SR_out.detach()[0])
                # cv2.imwrite(targetPath+"//"+file[0:-4]+"_lam.png", lam_SR_out_img[:, :, [2, 1, 0]])






