import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter

from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from src.data_util import get_palette, get_class_names
from PIL import Image


class pixel_classifier_v2(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier_v2, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(dim, dim//2, kernel_size=1),
            nn.BatchNorm2d(dim//2),
            nn.ReLU(),
            nn.Conv2d(dim//2, 1, kernel_size=1))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)


    def forward(self, x):
        return self.layers(x)



# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=128),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(32, numpy_class)
        )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        return self.layers(x)



def predict_labels_v2(models, images, size):
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(images)
    
    mean_seg = None
    seg_mode_ensemble = []

    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            
            preds = models[MODEL_NUMBER](images.cuda())
            preds = preds.view(preds.shape[2], preds.shape[3])
            # print("preds shape : ", preds.shape)

            if mean_seg is None:
                mean_seg = sigmoid(preds)
            else:
                mean_seg += sigmoid(preds)

            img_seg = (sigmoid(preds) >= 0.5).int()

            img_seg = img_seg.cpu().detach()
            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(models)

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]

        mean_seg = mean_seg.cpu().detach()
    return mean_seg, img_seg_final



def predict_labels(models, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []

    #TODO Softmax or Sigmoid

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            
            # For 256 x 256 image = 65536 pixels, preds will be [65536,2]
            preds = models[MODEL_NUMBER](features.cuda())

            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            # Probability map -> [65536,2]
            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            # Class 0 or 1 per pixel (using log softmax) -> [65536]
            img_seg = oht_to_scalar(preds)

            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()
            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        # Assign the label to the pixel that is the most frequent one across all models
        # [256, 256, num_models] -> [256, 256]
        img_seg_final = torch.mode(img_seg_final, 2)[0]

        # [65536,2] -> [65536,1] -> [256,256]
        mean_seg = mean_seg[:,1]
        mean_seg = mean_seg.reshape(*size)
        mean_seg = mean_seg.cpu().detach()
    return mean_seg, img_seg_final, top_k


def save_predictions(args, image_paths, preds, prob_maps, gts):
    # palette = get_palette(args['category'])
    # os.makedirs(os.path.join(args['exp_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], args['segmentations_folder']), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], args['ground_truths_folder']), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], args['probability_maps_folder']), exist_ok=True)

    for i, (pred, gt, prob_map) in enumerate(zip(preds, gts, prob_maps)):
        filename = image_paths[i].split('/')[-1].split('.')[0]
        # pred = np.squeeze(pred)
        # np.save(os.path.join(args['exp_dir'], 'predictions', filename + '.npy'), pred)
        # Save Segmented Image
        seg_img = Image.fromarray(pred.astype('uint8') * 255)
        seg_img.save(os.path.join(args['exp_dir'], args['segmentations_folder'], filename + '.png'))

        # Save the ground truth
        gt = np.squeeze(gt)
        gt_img = Image.fromarray(gt.astype('uint8') * 255)
        gt_img.save(os.path.join(args['exp_dir'], args['ground_truths_folder'], filename + '.png'))

        # Save the probability map
        prob_map = np.squeeze(prob_map)
        prob_map_img = Image.fromarray((prob_map * 255.0).astype('uint8'))
        prob_map_img.save(os.path.join(args['exp_dir'], args['probability_maps_folder'], filename + '.png'))


def compute_iou(args, preds, gts, print_per_class_ious=True):
    class_names = get_class_names(args['category'])

    ids = range(args['number_class'])

    unions = Counter()
    intersections = Counter()

    for pred, gt in zip(preds, gts):
        for target_num in ids:
            if target_num == args['ignore_label']: 
                continue
            preds_tmp = (pred == target_num).astype(int)
            gts_tmp = (gt == target_num).astype(int)
            unions[target_num] += (preds_tmp | gts_tmp).sum()
            intersections[target_num] += (preds_tmp & gts_tmp).sum()
    
    ious = []
    for target_num in ids:
        if target_num == args['ignore_label']: 
            continue
        iou = intersections[target_num] / (1e-8 + unions[target_num])
        ious.append(iou)
        if print_per_class_ious:
            print(f"IOU for {class_names[target_num]} {iou:.4}")
    return np.array(ious).mean()


def load_ensemble(args, device='cpu'):
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']
        model = nn.DataParallel(pixel_classifier(args["number_class"], args['dim'][-1]))
        model.load_state_dict(state_dict)
        model = model.module.to(device)
        models.append(model.eval())
    return models


def load_ensemble_v2(args, device='cpu'):
    models = []
    for i in range(args['model_num']):
        model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']
        model = nn.DataParallel(pixel_classifier_v2(numpy_class=(args['number_class']), dim=args['dim'][-1]))
        model.load_state_dict(state_dict)
        model = model.module.to(device)
        models.append(model.eval())
    return models