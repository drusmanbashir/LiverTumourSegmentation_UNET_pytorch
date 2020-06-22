import argparse,json
import  time, os, pkbar, torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from loss import dice_loss
from unet import UNet
from utils import  dsc
from torch.utils.tensorboard import SummaryWriter
import torchvision
from collections import defaultdict

def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss
# from torchvision import datasets, transforms

def makedirs(args):
    weights_fldr=args.save_weights.parent

    os.makedirs(weights_fldr, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    args.weights = args.weights._str   #POSIX path to string
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)

def get_batch_dsc(y_pred_batch,mask):
        batch_dsc = []
        batch_size = mask.shape[0]
        for case in range(batch_size):
            y_pred = y_pred_batch[case, 0, :]  # get a 3d volume for each case
            y_true = mask[case, 0, :]
            batch_dsc.append(dsc(y_pred, y_true))
        return batch_dsc


def multiclass_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss




def main(args, loader_train, loader_valid):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    batch_size = args.batch_size
    init_features=args.init_features
    load_weights_path=args.load_weights
    save_weights_path=args.save_weights
    total_train, total_val = args.num_cases
    loaders = {"train": loader_train, "valid": loader_valid}
    writer = SummaryWriter(args.writer_path)

    unet = UNet(in_channels=1, out_channels=args.num_classes,init_features=init_features)
    unet.to(device)
    # unet.load_state_dict(torch.load(args.load_weights))
    # bs = args.batch_size

    best_validation_dsc = 0.2

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    # loss_train = []
    # loss_valid = []
    # all_epochs_mean_dsc=[]
    #
    # validation_pred_batch = np.zeros(args.batch_shape)# i.e.k num_samples, channels, slices, w, h
    start_time=time.time()
    kbar_perepoch = pkbar.Kbar(target=args.epochs, width=28)
    running_loss_train=0
    for epoch in range(3,15):
        epoch_start = time.time()
        for phase in ["train", "valid"]:
            print(phase)
            if phase == "train":
                unet.train()
                step = 0
            else:
                unet.eval()
            metrics = defaultdict(float)
            dataset_size = len(loaders[phase])
            kbar_perbatch = pkbar.Kbar(target=dataset_size, width=28)  # i.e., number of slices
            validation_pred = []
            validation_true = []
            dsc_valid_epoch=[]
            # dsc_train_dataset =[] # list of dice losses, one entry per batch
            batch_dsc=0.0
            for i, sample in enumerate(loaders[phase]):
                    kbar_perbatch.update(i, values=[("Mean batch Dice Index",np.round(np.median(batch_dsc),2)), ("Std dev.",np.round(np.std(batch_dsc),1))])
                    im_3d , mask = sample
                    im_3s = im_3d.to(device)
                    y_pred_batch=torch.zeros(im_3s.shape)
                    mask=mask.to(device)
                    for s in range(im_3s.shape[4]): #one slice from each casein a batch at a time
                            imgs= im_3s[:,:,:,:,s]
                            msk=mask[:,:,:,:,s]
                            with torch.set_grad_enabled(phase == "train"):
                                optimizer.zero_grad()
                                y_pred = unet(imgs)
                                loss = calc_loss(y_pred, msk,metrics)
                                # loss2 = DiceLoss2(y_pred.to('cpu'),num_classes=1, targets=msk.to('cpu'))
                                if phase == "train":
                                    loss.backward()
                                    optimizer.step()
                                    # running_loss_train+=loss.item()*batch_size
                                y_pred_batch[:, :, :,:,s] = y_pred.data#.cpu()  # reverse transpose above


                    if phase == "train" and (s+1) % batch_size*2 == 0:
                                writer.add_scalar('Training loss',
                                                  loss.item(),
                                                  epoch*dataset_size+i)
                                writer.add_scalar('BCE',
                                                  metrics['bce'],
                                                  epoch*dataset_size+i)
                                writer.add_scalar('DICE',
                                                  metrics['dice'],
                                                  epoch*dataset_size+i)
                                writer.add_scalar('Metrics-loss',
                                                  metrics['loss'],
                                                  epoch*dataset_size+i)
                                im_pred = y_pred_batch[:, :, 50, :]
                                img_grid = torchvision.utils.make_grid(im_pred)
                                writer.add_image('16_seg_images', img_grid, epoch*dataset_size+i)
                                im_tru = mask[:,:,50,:]
                                img_grid2 = torchvision.utils.make_grid(im_tru)
                                writer.add_image('16_true masks', img_grid2, epoch*dataset_size+i)
                                running_loss = 0.0

                    if phase == 'valid':   #  once all slices of validation_pred_batch have been filled for the BATCH,
                        # print("Computing validation batch dice index..please wait")
                        batch_dsc = get_batch_dsc(y_pred_batch.cpu().numpy(), mask.cpu().numpy())
                        mean_valid_batch_dsc = np.mean(batch_dsc)
                        writer.add_scalar("Dice index validation, mean",mean_valid_batch_dsc,epoch*dataset_size+i)
                        writer.add_scalar("Dice index validation, std", np.std(batch_dsc), epoch * dataset_size + i)

# at the end of each epoch we get dice loss of all samples
        # mean_train_dsc= np.mean(dsc_train_dataset)
        if phase=='valid':
            if  mean_valid_batch_dsc > best_validation_dsc:
                best_validation_dsc = mean_valid_batch_dsc
                print('Saving Mean dice best unet weights',mean_valid_batch_dsc)
                torch.save(unet.state_dict(),args.save_weights)
        #=== update information at the end of epoch
        kbar_perepoch.update(i,values=[("Average validation dataset Dice Loss : ", mean_valid_batch_dsc)])
        time_since_epoch = divmod(time.time() - epoch_start, 60)
        time_since_start = divmod(time.time() - start_time, 60)
        # print_string = "Average training loss is {0:.2f}, Validation dice index is {1:.2f}, at the end of epoch {2}, time in epoch: {3}min {4:.0f} sec,overall time: {5}min {6:.0f}sec".format(np.mean(running_loss),mean_dsc_valid_dataset, epoch + 1,
        #                                                                                                                                                         time_since_epoch[0], time_since_epoch[1], time_since_start[0], time_since_start[1])
        # print(print_string)
        #All batches have been processed for this ep
        # validation_pred.append(validation_pred_batch)  # add this batch of cases to the list of all batches from the loader
        # validation_true.append(mask.cpu().numpy())
        #         # log_loss_summary(logger, loss_valid, step, prefix="val_")
        #         # logger.scalar_summary("val_dsc", mean_dsc, step)
        #
        # all_epochs_mean_dsc.append(mean_valid_batch_dsc)
                # loss_valid = []

    writer.close()
    #     print('Mean dsc loss :',loss_train)
    # print("Best validation mean DSC: {:4f}".format(best_validation_dsc))
    # return loss_train


for i, sample in enumerate(loaders[phase]):
    print(i)
    a,b = sample
    print(a.shape)


