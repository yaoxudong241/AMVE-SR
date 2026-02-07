import torch
from model.layers import *
import torch.nn.functional as F
import utils
import torch.nn as nn
import torchvision

def forward_chop(model, x, shave=6, min_size=10000,scale =2):
    # scale = 2#self.scale[self.idx_scale]
    n_GPUs = 1#min(self.n_GPUs, 4)
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = model(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(model, patch, shave=shave, min_size=min_size, scale= scale) \
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    output = x.new(b, c, h, w)
    output[:, :, 0:h_half, 0:w_half] \
        = sr_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

def reduce_func(method):
    """

    :param method: ['mean', 'sum', 'max', 'min', 'count', 'std']
    :return:
    """
    if method == 'sum':
        return torch.sum
    elif method == 'mean':
        return torch.mean
    elif method == 'count':
        return lambda x: sum(x.size())
    else:
        raise NotImplementedError()


def attr_grad(tensor, h, w, window=8, reduce='sum'):
    """
    :param tensor: B, C, H, W tensor
    :param h: h position
    :param w: w position
    :param window: size of window
    :param reduce: reduce method, ['mean', 'sum', 'max', 'min']
    :return:
    """
    h_x = tensor.size()[2]
    w_x = tensor.size()[3]
    h_grad = torch.pow(tensor[:, :, :h_x - 1, :] - tensor[:, :, 1:, :], 2)
    w_grad = torch.pow(tensor[:, :, :, :w_x - 1] - tensor[:, :, :, 1:], 2)
    grad = torch.pow(h_grad[:, :, :, :-1] + w_grad[:, :, :-1, :], 1 / 2)
    crop = grad[:, :, h: h + window, w: w + window]
    # return reduce_func(reduce)(crop)
    return crop

def attribution_objective(attr_func, h, w, window=16):
    def calculate_objective(image):
        return attr_func(image, h, w, window=window)
    return calculate_objective

class SR_CAM:
    def __init__(self,model,base_cam):
        self.model = model
        self.base_cam = base_cam

    def svd(self,I):
        I = torch.nan_to_num(I[0])
        reshaped_I = (I).reshape(
                I.shape[0], -1)
        reshaped_I= reshaped_I - reshaped_I.mean(dim=1)[:,None]
        U, S, VT = torch.linalg.svd(reshaped_I, full_matrices=False)
        d = int(S.shape[0] * 0.1)
        s = torch.diag(S[:d],0)
        new_I = U[:,:d].mm(s).mm(VT[:d,:])
        new_I = new_I.reshape(I.size())
        return new_I

    def find_last_layer(self):
        if isinstance(self.model,VGG):
            return self.model.features[-1]
    
    def get_weight_by_grad_cam(self,input, SR, HR, subimg_x, subimg_y, subimg_size,layer_name):
        value = dict()
        def backward_hook(module, grad_input, grad_output):
            value["gradients"] = grad_output[0]
        def forward_hook(module, input, output):
            value["activations"] = output

        l1_criterion = nn.L1Loss()

        all_modules = list(self.model.named_modules())
        leaf_modules = [(name, module) for name, module in all_modules if not list(module.children())]
        count = len(leaf_modules)
        for name, module in reversed(leaf_modules):

        # for (name, module) in self.model.named_modules().children():

            if name == layer_name:
                print("lastlayername:", name, ",Module type:", type(module))


                h1=module.register_forward_hook(forward_hook)
                h2=module.register_backward_hook(backward_hook)

                SR_out = self.model(input)

                # 边缘损失
                SR_patch = SR_out[:, :, subimg_x: subimg_x + subimg_size, subimg_y: subimg_y + subimg_size]
                HR_patch = HR[:, :, subimg_x: subimg_x + subimg_size, subimg_y: subimg_y + subimg_size]
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

                # loss = l1_criterion(edge_labels * SR_patch,
                #                           edge_labels * HR_patch)

                # loss = l1_criterion(SR_patch,
                #                           HR_patch)


                SR_padded = F.pad(SR_patch, (1, 1, 1, 1), mode='replicate')
                grad_x_SR = F.conv2d(SR_padded, sobel_x, groups=3)
                grad_y_SR = F.conv2d(SR_padded, sobel_y, groups=3)

                edge_SRs = torch.sqrt(grad_x_SR ** 2 + grad_y_SR ** 2)

                loss = (edge_SRs).sum()


                loss.backward()

                h1.remove()
                h2.remove()
                break
        return value["gradients"],value["activations"]
    
    def get_weight_by_score_cam(self,input,SR, HR,subimg_x, subimg_y, subimg_size,layer):
        value = dict()
        def forward_hook(module, input, output):
            value["activations"] = output
        h=layer.register_forward_hook(forward_hook)

        with torch.no_grad():
            self.model(input)
            h.remove()
            activations = value["activations"]
            weight = None
            batch = 8
            saliency_map = F.interpolate(activations, size=(HR.size()[-2],HR.size()[-1]), mode='bilinear', align_corners=False)
            saliency_map = torch.nan_to_num(saliency_map)
            maxs = saliency_map.view(saliency_map.size(0),
                                        saliency_map.size(1), -1).max(dim=-1)[0]
            mins = saliency_map.view(saliency_map.size(0),
                                        saliency_map.size(1), -1).min(dim=-1)[0]
            eps = torch.where(maxs==0,1e-9,0.0)
            saliency_map = (saliency_map - mins[:,:,None,None])/(maxs[:,:,None,None]-mins[:,:,None,None]+eps[:,:,None,None])
            saliency_map = saliency_map[0]

            for i in range(0,saliency_map.size(0),batch):
                x = input * saliency_map[i:i+batch,None,:,:]
                SR_out =self.model(x)

                y = utils.compute_psnr(HR, SR_out)
                if i==0:
                    weight = y.clone()
                else:
                    weight = torch.cat([weight,y])
            return weight,activations


    def get_explanation_component(self,input, SR, HR, subimg_x, subimg_y, subimg_size, layer_name=None):

        if self.base_cam.lower() == 'grad_cam':
            weight,activation = self.get_weight_by_grad_cam(input,SR, HR,subimg_x, subimg_y, subimg_size,layer_name)

            # for temp in range(weight.size(1)):
            #
            #     torchvision.utils.save_image(((weight-weight.min())/(weight.max()-weight.min()))[:,temp,:,:], f'output_images/image_{temp}.png')
            #     torchvision.utils.save_image(activation[:, temp, :, :]*10 , f'output_images/feature_{temp}.png')
            #     torchvision.utils.save_image(((weight-weight.min())/(weight.max()-weight.min()))[:,temp,:,:]*activation[:, temp, :, :] , f'output_images/exp_{temp}.png')

            # weight = torch.mean(weight, dim=(2, 3), keepdim=True)
            # weight = torch.relu(weight)
            I = weight*activation
            I = torch.relu(I)
            # print("I.max()", I.max())
        if self.base_cam.lower() == 'score_cam':
            weight,activation = self.get_weight_by_score_cam(input,SR, HR,subimg_x, subimg_y, subimg_size,layer_name)
            I = weight[None,:,None,None]*activation

        return I

    
    def forward(self, input, HR, subimg_x, subimg_y, subimg_size, layer_name, target_layer):

        SR = 0

        I = self.get_explanation_component(input,SR, HR, subimg_x, subimg_y, subimg_size, layer_name)

        # I = (torch.abs(HR / SR - 1))
        # # # 创建一个全零张量，形状为 (1, 3, 600, 600)
        # # zero_tensor = torch.zeros_like(I)
        # #
        # # # 将 (x: 30-50, y: 70-80) 的区域从原张量复制到全零张量中
        # # zero_tensor[:, :, subimg_x:subimg_x+subimg_size,subimg_y:subimg_y+subimg_size] = I[:, :, subimg_x:subimg_x+subimg_size,subimg_y:subimg_y+subimg_size]
        # # I = zero_tensor+ 0.000001

        for name, module in self.model.named_modules():

            if not list(module.children()):
                module.register_forward_hook(forward_hook)

        self.model(input)
        self.model.remove_hook()

        all_modules = list(self.model.named_modules())
        # 过滤出最底层的模块
        leaf_modules = [(name, module) for name, module in all_modules if not list(module.children())]

        # I = self.model.improve_resolution(I,target_layer)

        count = len(leaf_modules)

        # I=self.svd(I)
        # print("I:", I.max())


        for name, module in reversed(leaf_modules):

            count = count - 1
            if (count <= (len(leaf_modules))):
            # if (count >= target_layer) & (count <= (len(leaf_modules) - 4)):
                temp = module
                print(f"Layer Name: {name}, Layer: {module}")
                if hasattr(temp, 'IR'):
                    I = temp.IR(I)
                else:
                    continue
                if "head.0" in name:
                    head0_I = I
                    break
                elif "tail.1" in name:
                    tail1_I = I
                elif "body.5" in name:
                    print("body.5_name",name)
                    body5_I = I
                elif "body.10" in name:
                    body10_I = I
                    print("body.10_name", name)

        head0_I = torch.sum(head0_I,dim=1)
        tail1_I = torch.sum(tail1_I, dim=1)
        body5_I = torch.sum(body5_I, dim=1)
        body10_I = torch.sum(body10_I, dim=1)

        return head0_I, body5_I, body10_I, tail1_I

    def __call__(self,input, HR, subimg_x, subimg_y, subimg_size, layer_name, target_layer):
        return self.forward(input, HR, subimg_x, subimg_y, subimg_size, layer_name, target_layer)