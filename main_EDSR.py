import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
from scipy import stats
import argparse
import util
from SR_CAM import SR_CAM
from PIL import Image
import model.EDSR
import utils
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

    parser.add_argument("--HRPath", type=str, default=r'data\HR',
                        help='the path of HR images to be interpreted')

    parser.add_argument("--LRPath", type=str, default=r'data\LR',
                        help='the path of LR images to be interpreted')

    parser.add_argument("--savePath", type=str, default=r'data',
                        help='the path of interpretation maps')

    parser.add_argument("--checkpoint", type=str, default=r'weights/epoch_1000.pth',
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

# def main(opts):
def normalize_and_expand_channel(arr):
    Z = arr / arr.max()
    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('Purples')
    map_color = (255 * cmap(Z * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(Z)).astype(np.uint8)
    # 1. 拉伸到 0-255 之间
    # arr_min, arr_max = arr.min(), arr.max()
    # norm_arr = (arr - arr_min) / (arr_max - arr_min) * 255
    # norm_arr = norm_arr.astype(np.uint8)  # 转为 uint8 类型
    #
    # # 2. 创建一个 (512, 512, 3) 的 RGB 数组
    # rgb_array = np.zeros((512, 512, 3), dtype=np.uint8)
    #
    # # 3. 将红色通道赋值为拉伸后的数组
    # rgb_array[:, :, 2] = norm_arr
    #
    # # 4. 将绿色和蓝色通道赋值为 255
    # rgb_array[:, :, 1] = 0
    # rgb_array[:, :, 0] = 0

    # 5. 转换为 PIL 图像并返回
    return map_color
def vis_saliency_kde(map, zoomin=4):
    grad_flat = map.reshape((-1))
    datapoint_y, datapoint_x = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    Y, X = np.mgrid[0:map.shape[0]:1, 0:map.shape[1]:1]
    positions = np.vstack([X.ravel(), Y.ravel()])
    pixels = np.vstack([datapoint_x.ravel(), datapoint_y.ravel()])
    kernel = stats.gaussian_kde(pixels, weights=grad_flat)
    Z = np.reshape(kernel(positions).T, map.shape)
    Z = Z / Z.max()
    cmap = plt.get_cmap('seismic')
    # cmap = plt.get_cmap('Purples')
    map_color = (255 * cmap(Z * 0.5 + 0.5)).astype(np.uint8)
    # map_color = (255 * cmap(Z)).astype(np.uint8)
    # Img = Image.fromarray(map_color)
    # s1, s2 = Img.size
    # return Img.resize((s1 * zoomin, s2 * zoomin), Image.BICUBIC)
    return map_color

if __name__ == '__main__':
    # parameters of the pre-trained model
    scale_str = "8"
    seed_str = "176"

    opts = get_argparser().parse_args()
    print(opts.checkpoint)

    model = model.EDSR.EDSR(opts)  # pre-trained model
    model.load_state_dict(torch.load(opts.checkpoint), strict=True)  # True)
    model.eval()

    cuda = opts.cuda
    device = torch.device('cuda' if cuda else 'cpu')
    if cuda:
        model = model.to(device)

    SR_CAM = SR_CAM(model, opts.base_cam)

    # the path of HR images to be interpreted
    paths = os.listdir(opts.HRPath)
    for path in paths:
        # the path of HR images to be interpreted
        imname = (opts.HRPath +r'/{}').format(path)
        # the save path of interpretation maps
        save_path = (opts.savePath + r'/{}/{}').format("SR_cam",
                                             (imname.split('/')[-1]))
        im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
        im_gt2 = im_gt
        im_gt = utils.modcrop(im_gt, opts.upscale_factor)
        # the path of LR images to be interpreted
        im_l = cv2.imread((opts.LRPath + r'/X{}/{}').format(scale_str,path), cv2.IMREAD_COLOR)[:,
               :, [2, 1, 0]]  # BGR to RGB
        im_l2 = im_l
        if len(im_gt.shape) < 3:
            im_gt = im_gt[..., np.newaxis]
            im_gt = np.concatenate([im_gt] * 3, 2)
            im_l = im_l[..., np.newaxis]
            im_l = np.concatenate([im_l] * 3, 2)
        im_input = im_l / 255.0
        im_input = np.transpose(im_input, (2, 0, 1))
        im_input = im_input[np.newaxis, ...]
        im_input = torch.from_numpy(im_input).float()

        im_gt = im_gt/255.0
        im_gt = np.transpose(im_gt, (2, 0, 1))
        im_gt = im_gt[np.newaxis, ...]
        im_gt = torch.from_numpy(im_gt).float()

        if cuda:
            im_input = im_input.to(device)
            im_gt = im_gt.to(device)

        # interpretation maps of some example layers
        explanation_head0, explanation_body5, explanation_body10, explanation_tail1= SR_CAM(im_input, im_gt, opts.subimage_x, opts.subimage_y, opts.subimage_size, opts.layer_name, opts.target_layer)

        # linear stretch to 0-1
        explanation1_head0,  heatmap_head0, explanation1_ori_head0,  = util.visual_explanation(explanation_head0, (im_gt.size()[-2],im_gt.size()[-1]))
        explanation1_body5,  heatmap_body5, explanation1_ori_body5,  = util.visual_explanation(explanation_body5, (im_gt.size()[-2], im_gt.size()[-1]))
        explanation1_body10, heatmap_body10, explanation1_ori_body10, = util.visual_explanation(explanation_body10, (im_gt.size()[-2], im_gt.size()[-1]))
        explanation1_tail1,  heatmap_tail1, explanation1_ori_tail1,  = util.visual_explanation(explanation_tail1, (im_gt.size()[-2], im_gt.size()[-1]))

        save_path = save_path[0:-4] +".png"

        # print(save_path)
        vis_exp = normalize_and_expand_channel(explanation1_ori_head0)[:,:,0:3]
        lamtyperesult = vis_exp[..., ::-1]*0.5+im_l2[..., ::-1]*0.5

        # overlay of original image and interpretation image at input layer
        cv2.imwrite(save_path[0:-4] +"_"+str(opts.target_layer) +"lam_type.png", lamtyperesult)

        plt.figure(figsize=(7, 4))
        c = 2
        if opts.target_layer != -1:
            c = 2

        plt.subplot(1, c, 1)

        rect_y = int(opts.subimage_x)
        rect_x = int(opts.subimage_y)
        rect_height = int(opts.subimage_size)
        rect_width = int(opts.subimage_size)

        im_gt2[rect_y:rect_y + rect_height, rect_x:rect_x + 3] = [255, 0, 0]  # 左边框
        im_gt2[rect_y:rect_y + rect_height, rect_x + rect_width - 3:rect_x + rect_width] = [255, 0, 0]  # 右边框
        im_gt2[rect_y:rect_y + 3, rect_x:rect_x + rect_width] = [255, 0, 0]  # 上边框
        im_gt2[rect_y + rect_height - 3:rect_y + rect_height, rect_x:rect_x + rect_width] = [255, 0, 0]  # 下边框

        plt.imsave(save_path[0:-4] +"_"+str(opts.layer_name) +"_rect.png",im_gt2)

        plt.subplot(1, c, 2)

        # interpretation maps of example layers
        plt.imsave(save_path[0:-4] + "_" + "head0" + "_vis.png", explanation1_ori_head0)
        plt.imsave(save_path[0:-4] + "_" + "_body5" + "_vis.png", explanation1_ori_body5)
        plt.imsave(save_path[0:-4] + "_" + "_body10" + "_vis.png", explanation1_ori_body10)
        plt.imsave(save_path[0:-4] + "_" + "_tail1" + "_vis.png", explanation1_ori_tail1)


    print("Done")
