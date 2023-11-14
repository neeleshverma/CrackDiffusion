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
from src.pixel_classifier import  load_ensemble, compute_iou, predict_labels, save_predictions, save_predictions, pixel_classifier
from src.datasets import ImageLabelDataset, FeatureDataset, make_transform
from src.feature_extractors import create_feature_extractor, collect_features

from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
from guided_diffusion.guided_diffusion.dist_util import dev


feature_extractor = None
args = None

def extract_image_features(batch_img, batch_label, noise, img_name, args):
    X = torch.zeros((batch_img.shape[0], *args['dim'][::-1]), dtype=torch.float)
    y = torch.zeros((batch_img.shape[0], *args['dim'][:-1]), dtype=torch.uint8)
    
    for i in range(batch_img.shape[0]):
        img = batch_img[i]
        label = batch_label[i]
        img = img[None].to(dev())
        features = feature_extractor(img, noise=noise)
        # # print("Features shape : ", features[0].shape)
        # for j in range(len(features)):
        #     print("Features : ", features[j].shape)

        # exit(0)
        X[i] = collect_features(args, features).cpu()
        for target in range(args['number_class']):
            if target == args['ignore_label']:
                continue
            if 0 < (label == target).sum() < 20:
                print(f'Delete small annotation from image {img_name} | label {target}')
                label[label == target] = args['ignore_label']
        y[i] = label

    print("X shape : ", X.shape)
    print("y shape : ", y.shape)
    exit(0)
    d = X.shape[1]
    X = X.permute(1,0,2,3).reshape(d, -1).permute(1, 0)
    y = y.flatten()
    return X[y != args['ignore_label']], y[y != args['ignore_label']]


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


def evaluation(args, models):
    print("")
    print("************************************* Evaluation ****************************************")
    eval_feature_extractor = create_feature_extractor(**args)
    dataset = prepare_dataset(args['testing_path'], args['image_size'], args['testing_number'], args['model_type'])
    noise = prepare_noise(args)

    preds, gts, uncertainty_scores, prob_maps = [], [], [], []
    for img, label in tqdm(dataset):
        img = img[None].to(dev())
        features = eval_feature_extractor(img, noise=noise)
        features = collect_features(args, features)
        # exit(0)
        x = features.view(args['dim'][-1], -1).permute(1, 0)

        prob_map, pred, uncertainty_score = predict_labels(
            models, x, size=args['dim'][:-1]
        )
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
    



# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def train(args):
    dataset = prepare_dataset(args['training_path'], args['image_size'], args['training_number'], args['model_type'])
    noise = prepare_noise(args)

    image_dataloader = DataLoader(dataset, batch_size=args['image_batch_size'], shuffle=True)

    global feature_extractor
    feature_extractor = create_feature_extractor(**args)

    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):
        gc.collect()
        classifier = pixel_classifier(numpy_class=(args['number_class']), dim=args['dim'][-1])
        classifier.init_weights()

        classifier = nn.DataParallel(classifier).cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()

        # iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0
        flag = 0

        for epoch in range(args['num_epochs']):
            print("")
            print("******************************** EPOCH {} **********************************".format(epoch))
            iteration = 0
            for row, (batch_img, batch_label) in enumerate(image_dataloader):
                features, labels = extract_image_features(batch_img, batch_label, noise, dataset.image_paths[row], args)

                train_data = FeatureDataset(features, labels)
                train_loader = DataLoader(dataset=train_data, batch_size=args['feature_batch_size'], shuffle=True, drop_last=True)
                
                for X_batch, y_batch in train_loader:
                    
                    X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())
                    y_batch = y_batch.type(torch.long)

                    optimizer.zero_grad()
                    y_pred = classifier(X_batch)
                    print("X batch : ", X_batch.shape)
                    print("y pred : ", y_pred.shape)
                    exit(0)
                    loss = criterion(y_pred, y_batch)
                    acc = multi_acc(y_pred, y_batch)

                    loss.backward()
                    optimizer.step()

                    iteration += 1
                    if iteration % 1000 == 0:
                        print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item(), 'acc', acc)
                    if epoch > 3:
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            break_count = 0
                        else:
                            break_count += 1

                        if break_count > 50:
                            stop_sign = 1
                            print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch), "***************")
                            break

                if stop_sign == 1:
                    flag = 1
                    break
            
            if flag == 1:
                break

        model_path = os.path.join(args['exp_dir'], 
                                  'model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:',model_path)
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)
    

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

    # Prepare the experiment folder 
    # if len(opts['steps']) > 0:
    #     suffix = '_'.join([str(step) for step in opts['steps']])
    #     suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
    #     opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

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
        train(opts)
    
    print('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda')
    evaluation(opts, models)
