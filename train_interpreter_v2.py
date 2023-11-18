import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os
import gc
import numpy as np

from torch.utils.data import DataLoader
from torchsummary import summary
import torchinfo

import argparse
from src.utils import setup_seed, multi_acc, get_prob_map_list, ods_metric, ois_metric
from src.pixel_classifier import  load_ensemble_v2, compute_iou, predict_labels_v2, save_predictions, pixel_classifier_v2
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform
from src.feature_extractors import create_feature_extractor, collect_features_v2

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev


feature_extractor = None
args = None

def extract_image_features_v2(batch_img, batch_label, noise, img_name, args):
    X = torch.zeros((batch_img.shape[0], args['dim'][-1], 256, 256), dtype=torch.float)
    y = torch.zeros((batch_img.shape[0], 256, 256), dtype=torch.uint8)

    for i in range(batch_img.shape[0]):
        img = batch_img[i]
        label = batch_label[i]
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)

        X[i] = collect_features_v2(args, features).cpu()
        for target in range(args['number_class']):
            if target == args['ignore_label']:
                continue
            if 0 < (label == target).sum() < 20:
                print(f'Delete small annotation from image {img_name} | label {target}')
                label[label == target] = args['ignore_label']
        y[i] = label

    return X, y


def prepare_dataset(data_dir, image_size, num_images, model_type):
    return ImageLabelDataset(
        data_dir=data_dir,
        resolution=image_size,
        num_images=num_images,
        transform=make_transform(
            model_type,
            image_size
        )
    )


def prepare_noise(args):
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, 3, args['image_size'], args['image_size'], 
                            generator=rnd_gen, device=dev())
    else:
        noise = None
    
    return noise


def evaluation_v2(args, models):
    print("")
    print("************************************* Evaluation ****************************************")
    eval_feature_extractor = create_feature_extractor(**args)
    dataset = prepare_dataset(args['testing_path'], args['image_size'], args['testing_number'], args['model_type'])
    noise = prepare_noise(args)

    preds, gts, uncertainty_scores, prob_maps = [], [], [], []
    for img, label in tqdm(dataset):
        img = img[None].to(dev())
        X = eval_feature_extractor(img, noise=noise)
        X = collect_features_v2(args, X)
        X = torch.unsqueeze(X, dim=0)  
        print("X shape : ", X.shape)

        prob_map, pred = predict_labels_v2(
            models, X, size=args['dim'][-1]
        )
        print(pred)
        print(label)
        exit(0)
        gts.append(label.numpy())
        preds.append(pred.numpy())
        prob_maps.append(prob_map.numpy())
        uncertainty_scores.append(uncertainty_score.item())
    
    save_predictions(args, dataset.image_paths, preds, prob_maps, gts)

    # Compute and print mean IoU
    miou = compute_iou(args, preds, gts)
    print("--------------------------------- IoU Scores ----------------------------------")
    print(f'Overall mIoU: ', miou)
    print(f'Mean uncertainty: {sum(uncertainty_scores) / len(uncertainty_scores)}')
    print("-------------------------------------------------------------------------------")

    log_txt = "Overall mIoU : {}\nMean uncertainty : {}\n\n".format(miou, sum(uncertainty_scores) / len(uncertainty_scores))

    # Compute and print ODS, OIS scores
    prob_maps_list, gt_list = get_prob_map_list(args)
    ods_score_list = ods_metric(prob_maps_list, gt_list)
    best_ods_score = np.amax(ods_score_list, axis=0)
    ois_score = ois_metric(prob_maps_list, gt_list)

    print("--------------------------------- ODS & OIS ----------------------------------")
    print(f'ODS Score : ', best_ods_score[3])
    print(f'OIS Score : ', ois_score)
    print("------------------------------------------------------------------------------")

    log_txt += "ODS Score : {}\n\nOIS Score : {}".format(best_ods_score[3], ois_score)
    log_file = os.path.join(args['exp_dir'], args['log_file'])

    with open(log_file, 'a', encoding='utf-8') as fout:
        fout.write(str(models[0]))
        fout.write("\n")
        fout.write(log_txt)
    

def train_v2(args):
    dataset = prepare_dataset(args['training_path'], args['image_size'], args['training_number'], args['model_type'])
    noise = prepare_noise(args)

    image_dataloader = DataLoader(dataset, batch_size=args['image_batch_size'], shuffle=True)

    global feature_extractor
    feature_extractor = create_feature_extractor(**args)

    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):
        gc.collect()
        classifier = pixel_classifier_v2(numpy_class=(args['number_class']), dim=args['dim'][-1])
        classifier.init_weights()

        classifier = nn.DataParallel(classifier).cuda()
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()


        for epoch in range(args['num_epochs']):
            epoch_loss = 0
            print("")
            print("******************************** EPOCH {} **********************************".format(epoch))
            iteration = 0
            for row, (batch_img, batch_label) in enumerate(image_dataloader):
                X_batch, y_batch = extract_image_features_v2(batch_img, batch_label, noise, dataset.image_paths[row], args)

                X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())
                y_batch = y_batch.type(torch.float)

                optimizer.zero_grad()
                y_pred = classifier(X_batch)
                y_pred = y_pred.squeeze(dim=1)
                y_pred = y_pred.type(torch.float)

                loss = criterion(y_pred, y_batch)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            print("***************** Epoch {} : Loss {:.2f}".format(epoch, epoch_loss))
        
        model_path = os.path.join(args['exp_dir'], 'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('')
        print('Saving Model:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()}, model_path)
        print('')
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_and_diffusion_defaults())

    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int,  default=0)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    # Check whether all models in ensemble are trained 
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'model_{i}.pth')) 
                  for i in range(opts['model_num'])]
              
    if not all(pretrained):
        # train all remaining models
        opts['start_model_num'] = sum(pretrained)
        train_v2(opts)
    
    # exit(0)
    print('Loading pretrained models...')
    models = load_ensemble_v2(opts, device='cuda')
    evaluation_v2(opts, models)
