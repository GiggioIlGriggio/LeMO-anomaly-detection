import random
import argparse
import json
import torch
from torch.utils.data import DataLoader
import os
from cnn.resnet import wide_resnet50_2 as wrn50_2
from cnn.resnet import resnet18 as res18
from cnn.efficientnet import EfficientNet as effnet
from cnn.vgg import vgg19_bn as vgg19

import datasets.mvtec as mvtec
from datasets.mvtec import MVTecDataset
from utils.metric import *
from utils.visualizer import * 

from utils.cfa import *
import torch.optim as optim
import warnings

import time

warnings.filterwarnings("ignore", category=UserWarning)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def parse_args():
    parser = argparse.ArgumentParser('LeMO configuration')
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str, default='./mvtec_result')
    parser.add_argument('--Rd', type=bool, default=False)
    parser.add_argument('--cnn', type=str, choices=['res18', 'wrn50_2', 'effnet-b5', 'vgg19'], default='wrn50_2')
    parser.add_argument('--size', type=int, choices=[224, 256], default=224)
    parser.add_argument('--gamma_c', type=int, default=1)
    parser.add_argument('--gamma_d', type=int, default=1)
    parser.add_argument('--sigmoid', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--amsgrad', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--memory_update', nargs='+', type=str, choices=["memory_parameter_update", "kmeans", "no"], default="memory_parameter_update")
    parser.add_argument('--loss', nargs='+', type=str, choices=["online", "cfa", "tripletcentral","NCENEW"], default="NCENEW")
    parser.add_argument('--positives', type=int, default=3)
    parser.add_argument('--negatives', type=int, default=3)
    parser.add_argument('--num_prototypes', type=int, default=10)
    parser.add_argument('--add_coord', type=str, choices=["0","-1"], default="-1")
    parser.add_argument('--avg_pooling', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--class_name', type=str, default='all')
    parser.add_argument('--validation_steps', action=argparse.BooleanOptionalAction, default=False)
    
    return parser.parse_args()
validation_steps= [1,2,4,6,8,12,16,20,25,30,35,40,50,60,70,80,100,120,140,170,200]

#script modified for running final tests
def run():
    seeds = [33] #, 33
    sigmoid =[False]
    #memory_update = ["memory_parameter_update","no", "kmeans"] # "memory_parameter_update","no", "kmeans"
    #loss = ["NCENEW", "cfa"]    #"email","NCE","online", "cfa", "tripletcentral","online_nologexp","NCENEW"
    
    args = parse_args()
    
    validation_steps = [] if not args.validation_steps else [1,2,4,6,8,12,16,20,25,30,35,40,50,60,70,80,100,120,140,170,200]
    print(validation_steps)
    for seed in seeds:
        for sig in sigmoid:
            for l in args.loss:
                for mem in args.memory_update:
                    random.seed(seed)
                    torch.manual_seed(seed)
                    if use_cuda:
                        torch.cuda.manual_seed_all(seed)
                    print("LOSS: ", l)
                    print("MEORYUPDATE: ", mem)
                    print("SIGMOID: ", sig)
                    class_names = mvtec.CLASS_NAMES if args.class_name == 'all' else [args.class_name]

                    total_roc_auc = []
                    total_pixel_roc_auc = []
                    total_pixel_pro_auc = []

                    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
                    fig_img_rocauc = ax[0]
                    fig_pixel_rocauc = ax[1]

                    
                    for class_name in class_names:
                        output_file_path = f'RES_{class_name}_{seed}_{sig}_{l}_{mem}_pooling{args.avg_pooling}.json'
                        if output_file_path in os.listdir(args.results_path):
                          print("File " + output_file_path + "ALREADY EXIST -> SKIP")
                          continue
                        results = {}
                        results[class_name] = {}
                        best_img_roc = -1
                        best_pxl_roc = -1
                        best_pxl_pro = -1
                        print(' ')
                        print('%s | newly initialized...' % class_name)

                        train_dataset    = MVTecDataset(dataset_path  = args.data_path, 
                                                        class_name    =     class_name, 
                                                        resize        =            256,
                                                        cropsize      =      args.size,
                                                        is_train      =           True,
                                                        wild_ver      =        args.Rd)

                        test_dataset     = MVTecDataset(dataset_path  = args.data_path, 
                                                        class_name    =     class_name, 
                                                        resize        =            256,
                                                        cropsize      =      args.size,
                                                        is_train      =          False,
                                                        wild_ver      =        args.Rd)

                        train_loader   = DataLoader(dataset         = train_dataset, 
                                                    batch_size      =             1, 
                                                    pin_memory      =          True,
                                                    shuffle         =          True,
                                                    drop_last       =          True,)

                        test_loader   =  DataLoader(dataset        =   test_dataset, 
                                                    batch_size     =              1, 
                                                    pin_memory     =           True,)


                        if args.cnn == 'wrn50_2':
                            model = wrn50_2(pretrained=True, progress=True)
                        elif args.cnn == 'res18':
                            model = res18(pretrained=True,  progress=True)
                        elif args.cnn == 'effnet-b5':
                            model = effnet.from_pretrained('efficientnet-b5')
                        elif args.cnn == 'vgg19':
                            model = vgg19(pretrained=True, progress=True)
                        model = model.to(device)
                        for param in model.parameters():
                            param.requires_grad = False
                        model.eval()

                        loss_fn = DSVDD(model, train_loader, args.cnn, args.gamma_c, args.gamma_d, sig, mem, l, args.positives, args.negatives, args.num_prototypes, "0" if sig else "-1", args.avg_pooling, device)
                        loss_fn = loss_fn.to(device)

                        epochs = 1
                        params = [{'params' : loss_fn.parameters()},]
                        optimizer     = optim.AdamW(params        = params, 
                                                    lr            = 1e-3,
                                                    weight_decay  = 5e-4,
                                                    amsgrad       = args.amsgrad )

                        for epoch in range(epochs):
                            r'TEST PHASE'
                            results[class_name][f"epoch_{epoch}"] = {}

                            count = 1
                            loss_fn.train()
                            print("INIZIO IL TRAINING")
                            with tqdm(train_loader, '%s -->'%(class_name)) as t:
                                for (x, _, _) in t:
                                    test_imgs = list()
                                    gt_mask_list = list()
                                    gt_list = list()
                                    heatmaps = None

                                    optimizer.zero_grad()

                                    x = x.to(device)
                                    p = model(x)

                                    loss, _, phi_p = loss_fn(p)
                                    t.set_postfix({"Loss": loss.item()})
                                    loss.backward()
                                    # Calculate the gradient magnitudes for each parameter in the model
                                    #gradients = [param.grad.norm().item() for param in loss_fn.parameters() if param.grad is not None]
                                    #print("MAGNITUDE: ",sum(gradients), len(gradients))
                                    if mem == "kmeans":
                                        loss_fn.kmeans_update(phi_p)

                                    optimizer.step()
                                    if count == len(train_loader)-1 or count in validation_steps:
                                        loss_fn.eval()
                                        encoder_times = []
                                        decoder_times = []
                                        for x_t, y_t, mask_t in test_loader:

                                            test_imgs.extend(x_t.cpu().detach().numpy())
                                            gt_list.extend(y_t.cpu().detach().numpy())
                                            gt_mask_list.extend(mask_t.cpu().detach().numpy())

                                            start_time = time.time()
                                            p = model(x_t.to(device))
                                            time_enc = time.time()
                                            _, score, _ = loss_fn(p)
                                            time_dec = time.time()

                                            encoder_times.append(time_enc - start_time)
                                            decoder_times.append(time_dec - time_enc)

                                            heatmap = score.cpu().detach()
                                            heatmap = torch.mean(heatmap, dim=1)
                                            heatmaps = torch.cat((heatmaps, heatmap), dim=0) if heatmaps is not None else heatmap
                                        print(f"mean time enc: {sum(encoder_times)/len(encoder_times)}")
                                        print(f"mean time dec: {sum(decoder_times)/len(decoder_times)}")
                                        loss_fn.train()

                                        heatmaps = upsample(heatmaps, size=x.size(2), mode='bilinear')
                                        heatmaps = gaussian_smooth(heatmaps, sigma=4)

                                        gt_mask = np.asarray(gt_mask_list)
                                        scores = rescale(heatmaps)

                                        threshold = get_threshold(gt_mask, scores)
                                        r'Image-level AUROC'
                                        fpr_i, tpr_i, img_roc_auc = cal_img_roc(scores, gt_list)
                                        #best_img_roc = img_roc_auc if img_roc_auc > best_img_roc else best_img_roc
                                        #fig_img_rocauc.plot(fpr, tpr, label='%s img_ROCAUC: %.3f' % (class_name, img_roc_auc))
                                        print(img_roc_auc)
                                        r'Pixel-level AUROC'
                                        fpr_p, tpr_p, per_pixel_rocauc = cal_pxl_roc(gt_mask, scores)
                                        #best_pxl_roc = per_pixel_rocauc if per_pixel_rocauc > best_pxl_roc else best_pxl_roc
                                        r'Pixel-level AUPRO'
                                        per_pixel_proauc = cal_pxl_pro(gt_mask, scores)
                                        #best_pxl_pro = per_pixel_proauc if per_pixel_proauc > best_pxl_pro else best_pxl_pro


                                        results[class_name][f"epoch_{epoch}"][f"iteration_{count}"] = {}
                                        #results[class_name][f"epoch_{epoch}"][f"iteration_{count}"]["fpr_i"] = fpr_i.tolist()
                                        #results[class_name][f"epoch_{epoch}"][f"iteration_{count}"]["tpr_i"] = tpr_i.tolist()
                                        results[class_name][f"epoch_{epoch}"][f"iteration_{count}"]["img_roc_auc"] = img_roc_auc.tolist()
                                        #results[class_name][f"epoch_{epoch}"][f"iteration_{count}"]["fpr_p"] = fpr_p.tolist()
                                        #results[class_name][f"epoch_{epoch}"][f"iteration_{count}"]["tpr_p"] = tpr_p.tolist()
                                        results[class_name][f"epoch_{epoch}"][f"iteration_{count}"]["per_pixel_rocauc"] = per_pixel_rocauc.tolist()
                                        results[class_name][f"epoch_{epoch}"][f"iteration_{count}"]["per_pixel_proauc"] = per_pixel_proauc.tolist()
                                        print(results)
                                    count += 1

                                """
                                print('[%d / %d]image ROCAUC: %.3f | best: %.3f'% (epoch, epochs, img_roc_auc, best_img_roc))
                                print('[%d / %d]pixel ROCAUC: %.3f | best: %.3f'% (epoch, epochs, per_pixel_rocauc, best_pxl_roc))
                                print('[%d / %d]pixel PROAUC: %.3f | best: %.3f'% (epoch, epochs, per_pixel_proauc, best_pxl_pro))
                                """
                        
                        with open(os.path.join(args.results_path,f'RES_{class_name}_{seed}_{sig}_{l}_{mem}_pooling{args.avg_pooling}.json'), 'w') as file:
                            json.dump(results, file)
                        print("FILE SAVED")
    
"""
        print('image ROCAUC: %.3f'% (best_img_roc))
        print('pixel ROCAUC: %.3f'% (best_pxl_roc))
        print('pixel ROCAUC: %.3f'% (best_pxl_pro))

        total_roc_auc.append(best_img_roc)
        total_pixel_roc_auc.append(best_pxl_roc)
        total_pixel_pro_auc.append(best_pxl_pro)

        fig_pixel_rocauc.plot(fpr, tpr, label='%s ROCAUC: %.3f' % (class_name, per_pixel_rocauc))
        save_dir = args.save_path + '/' + f'pictures_{args.cnn}'
        os.makedirs(save_dir, exist_ok=True)
        plot_fig(test_imgs, scores, gt_mask_list, threshold, save_dir, class_name)

    print('Average ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.title.set_text('Average image ROCAUC: %.3f' % np.mean(total_roc_auc))
    fig_img_rocauc.legend(loc="lower right")

    print('Average pixel ROCUAC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.title.set_text('Average pixel ROCAUC: %.3f' % np.mean(total_pixel_roc_auc))
    fig_pixel_rocauc.legend(loc="lower right")

    print('Average pixel PROUAC: %.3f' % np.mean(total_pixel_pro_auc))
    
    fig.tight_layout()
    fig.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)
"""
if __name__ == '__main__':
    run()
