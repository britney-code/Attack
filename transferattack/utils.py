import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import timm
import os
import torch.nn as nn
from torch_nets import tf_adv_inception_v3, tf_ens3_adv_inc_v3, tf_ens4_adv_inc_v3, tf_ens_adv_inc_res_v2
from torchvision import models
import pretrainedmodels 
img_height, img_width = 299, 299 # [224 224]
img_max, img_min = 1., 0

# cnn_model_paper = ['resnet18', 'resnet101', 'resnext50_32x4d', 'densenet121']
cnn_model_paper = ['inception_v3', 'inception_v4', 'inception_resnet_v2','resnet50', 'densenet121']
vit_model_paper = ['vit_base_patch16_224', 'pit_b_224', 'visformer_small']
adv_model_paper = ['tf_ens3_adv_inc_v3', 'tf_ens4_adv_inc_v3', 'tf_ens_adv_inc_res_v2']

cnn_model_pkg = ['vgg19', 'resnet18', 'resnet101',
                 'resnext50_32x4d', 'densenet121', 'mobilenet_v2']
vit_model_pkg = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                 'tnt_s_patch16_224', 'levit_256', 'convit_base', 'swin_tiny_patch4_window7_224']

tgr_vit_model_list = ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224', 'visformer_small',
                      'deit_base_distilled_patch16_224', 'tnt_s_patch16_224', 'levit_256', 'convit_base']

generation_target_classes = [24, 99, 245, 344, 471, 555, 661, 701, 802, 919]

class Permute(nn.Module):
    def __init__(self, permutation=[2, 1, 0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        return input[:, self.permutation]

class Normalize(nn.Module):

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class TfNormalize(nn.Module):

    def __init__(self, mean=0, std=1, mode='tensorflow'):
        """
        mode:
            'tensorflow':convert data from [0,1] to [-1,1]
            'torch':(input - mean) / std
        """
        super(TfNormalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


def load_pretrained_model(cnn_model=[], vit_model=[], adv_model=[]):
    for model_name in cnn_model:
        if model_name in timm.list_models():
            print('=> Loading model {} from timm.models for evaluation [CNN]'.format(model_name))
            yield model_name, timm.create_model(model_name, pretrained=True)
        elif model_name in pretrainedmodels.__dict__:
            print('=> Loading model {} from pretrainedmodels for evaluation [CNN]'.format(model_name))
            model = pretrainedmodels.__dict__[model_name](pretrained='imagenet')
        else:
            print('=> Loading model {} from torchvision.models for evaluation [CNN]'.format(model_name))
            yield model_name, models.__dict__[model_name](weights="DEFAULT")
        # yield model_name, models.__dict__[model_name](weights="DEFAULT")
        # yield model_name, models.__dict__[model_name](weights="IMAGENET1K_V1")
        
    for model_name in vit_model:
        print('=> Loading model {} from timm.models for evaluation [VIT]'.format(model_name))
        yield model_name, timm.create_model(model_name, pretrained=True)

    for model_name in adv_model:
        print('=> Loading model {} from torch_nets.models for evaluation [ADV]'.format(model_name))
        modelpath = os.path.join('./torch_nets_weight/', model_name + '.npy')
        if model_name == 'tf_adv_inception_v3':
            net = tf_adv_inception_v3
            model = net.KitModel(modelpath).eval().cuda()
        elif model_name == 'tf_ens3_adv_inc_v3':
            net = tf_ens3_adv_inc_v3
            model = net.KitModel(modelpath).eval().cuda()
        elif model_name == 'tf_ens4_adv_inc_v3':
            net = tf_ens4_adv_inc_v3
            model = net.KitModel(modelpath).eval().cuda()
        elif model_name == 'tf_ens_adv_inc_res_v2':
            net = tf_ens_adv_inc_res_v2
            model = net.KitModel(modelpath).eval().cuda()
        else:
            raise NotImplementedError
        yield model_name, model
        
class PreprocessingModel(nn.Module):
    def __init__(self, resize, mean, std):
        super(PreprocessingModel, self).__init__()
        self.resize = transforms.Resize(resize)
        self.normalize = Normalize(mean, std)

    def forward(self, x):
        return self.normalize(self.resize(x))

def model_soruce(model):
    if model.__class__.__module__.startswith('torchvision'):
        return "torchvision"
    elif model.__class__.__module__.startswith('timm'):
        return "timm"
    elif model.__class__.__module__.startswith('pretrainedmodels'):
        return "pretrainedmodels"
    else:
        return "adv"


def wrap_model(model):
    """
    Add normalization layer with mean and std in training configuration
    """
    flag = model_soruce(model)
    if flag == 'timm':
        """timm.models"""
        mean = model.default_cfg['mean']
        std = model.default_cfg['std']
        normalize = Normalize(mean, std)
        if model.default_cfg['architecture'] in cnn_model_paper:
            return nn.Sequential(
                normalize, model
            )
        
        elif model.default_cfg['architecture'] in vit_model_paper:
            return nn.Sequential(
                PreprocessingModel(224, mean, std), model
            )
        
        else:
            raise ValueError(f'Unknown architecture: {model.default_cfg["architecture"]}')
        
    elif flag == "torchvision":
        """torchvision.models"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = Normalize(mean, std)
        model_name = model.__class__.__name__.lower() 
        if any(name in model_name for name in ['vgg']):
            return nn.Sequential(
                PreprocessingModel(224, mean, std), model
            )
             
        else: 
             return nn.Sequential(
                normalize,
                model 
            )
             
    elif flag == "pretrainedmodels":
        # 从 `pretrainedmodels` 获取模型的标准化参数
        if hasattr(model, 'mean') and hasattr(model, 'std'):
            mean = model.mean
            std = model.std
        else:
            raise ValueError("mean and std need to harcraft!")
        normalize = Normalize(mean, std)
        model_name = model.__class__.__name__.lower()
        if any(name in model_name for name in ['vgg']):
            return nn.Sequential(
                PreprocessingModel(224, mean, std), model
            ) 
        
        else:
            return nn.Sequential(
                normalize,
                model 
            )
    else:
        '''adversarial trained models'''
        return nn.Sequential(
            TfNormalize('tensorflow'),
            model,
        )

def save_images(output_dir, adversaries, filenames):
    adversaries = (adversaries.detach().permute((0, 2, 3, 1)).cpu().numpy() * 255).astype(np.uint8)
    for i, filename in enumerate(filenames):
        Image.fromarray(adversaries[i]).save(os.path.join(output_dir, filename))


def clamp(x, x_min, x_max):
    return torch.min(torch.max(x, x_min), x_max)


class EnsembleModel(torch.nn.Module):
    def __init__(self, models, mode='mean'):
        super(EnsembleModel, self).__init__()
        self.device = next(models[0].parameters()).device
        for model in models:
            model.to(self.device)
        self.models = models
        self.softmax = torch.nn.Softmax(dim=1)
        self.type_name = 'ensemble'
        self.num_models = len(models)
        self.mode = mode

    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        if self.mode == 'mean':
            outputs = torch.mean(outputs, dim=0)
            return outputs
        elif self.mode == 'ind':
            return outputs
        else:
            raise NotImplementedError


# ILSVRC2012 随机选择 1000 张图片
class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, eval=False):
        self.targeted = targeted
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'labels.csv'))

        if eval:
            self.data_dir = output_dir
            # load images from output_dir, labels from input_dir/labels.csv
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print('=> Train mode: training on {}'.format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __len__(self):
        return len(self.f2l.keys())

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]
        assert isinstance(filename, str)
        filepath = os.path.join(self.data_dir, filename)
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        image = np.array(image).astype(np.float32) / 255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]

        return image, label, filename

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            f2l = {dev.iloc[i]['filename']: [dev.iloc[i]['label'], dev.iloc[i]['targeted_label']] for i in
                   range(len(dev))}
        else:
            f2l = {dev.iloc[i]['filename']: dev.iloc[i]['label']
                   for i in range(len(dev))}
        return f2l


# NIPS 1000 加载
class AdvImagetNet(torch.utils.data.Dataset):
    def __init__(self, input_dir=None, output_dir=None, targeted=False, eval=False):
        self.targeted = targeted
        self.data_dir = input_dir
        self.f2l = self.load_labels(os.path.join(self.data_dir, 'images.csv'))
        if eval:
            self.data_dir = output_dir
            # load images from output_dir, labels from input_dir/images.csv
            print('=> Eval mode: evaluating on {}'.format(self.data_dir))
        else:
            self.data_dir = os.path.join(self.data_dir, 'images')
            print('=> Train mode: training on {}'.format(self.data_dir))
            print('Save images to {}'.format(output_dir))

    def __getitem__(self, idx):
        filename = list(self.f2l.keys())[idx]
        ImageID = filename + '.png'
        assert isinstance(filename, str)
        filepath = os.path.join(self.data_dir, ImageID)
        image = Image.open(filepath)
        image = image.resize((img_height, img_width)).convert('RGB')
        image = np.array(image).astype(np.float32) / 255
        image = torch.from_numpy(image).permute(2, 0, 1)
        label = self.f2l[filename]
        return image, label, ImageID

    def __len__(self):
        return len(self.f2l.keys())

    def load_labels(self, file_name):
        dev = pd.read_csv(file_name)
        if self.targeted:
            f2l = {dev.iloc[i]['ImageId']: [dev.iloc[i]['TrueLabel'] - 1, dev.iloc[i]['TargetClass'] - 1] for i in
                   range(len(dev))}
        else:
            f2l = {dev.iloc[i]['ImageId']: dev.iloc[i]['TrueLabel'] - 1 for i in range(len(dev))}
        return f2l


if __name__ == '__main__':
    dataset = AdvDataset(input_dir='./data_targeted',
                         targeted=True, eval=False)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=False, num_workers=0)

    for i, (images, labels, filenames) in enumerate(dataloader):
        print(images.shape)
        print(labels)
        print(filenames)
        break
