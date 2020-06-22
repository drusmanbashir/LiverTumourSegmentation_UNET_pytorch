# import prep_data_gcp as prep  #conflicts with opening plt imageshow
import platform, os,csv, re, argparse
import matplotlib
matplotlib.use('Tkagg')
# os.environ['KMP_DUPyyLICATE_LIB_OK']='True'# to avoid sigseg
import matplotlib.pyplot as plt
from prep_data_gcp import *
import pandas as pd
import numpy as np
# import nibabel as nib
from pathlib import Path
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 200)
import torch
import json

# from train_ub import main as main_tr
#
from torchvision import transforms
import importlib.util
if platform.system()=='Darwin':  # macbook vs gcp
    root = '/Users/ub'
    specs = importlib.util.spec_from_file_location("utils", "/Users/ub/Dropbox/code/utils/utils_2d.py")
    utils_2d= importlib.util.module_from_spec(specs)
    specs.loader.exec_module(utils_2d)
    main = Path(root) / 'datasets/MICCAI_BraTS_2019_Data_Training'
    all = Path(main) / 'all_tensor_segmentation.csv'
else:
    root = '/home/ub'
    import utils_2d
    main= Path(root)/'datasets/MICCAI_BraTS_2019_Data_Training'
    all = Path(main) / 'all_tensor_segmentation.csv'

df  = pd.read_csv(all)
row = df.iloc[0]
fl = torch.load(Path(root)/row.flair_filename)
msk = torch.load(Path(root)/row.seg_filename)
t1 =torch.load(Path(root)/row.t1ce_filename)

sample = []

lg= Path(root)/'datasets/MICCAI_BraTS_2019_Data_Training/LGG/'
hg= Path(root)/'datasets/MICCAI_BraTS_2019_Data_Training/HGG/'

# ========= SEGMENTATION ===============================

composed = transforms.Compose([RandomFlip('unet'),Rescale_tensor_unet(128)])
# composed = transforms.Compose([Rescale_tensor_unet(128)])
rescale_only= transforms.Compose([Rescale_tensor_unet(128)])
init_features =32
writer_path='runs/unet_16'  #location to save log files in
weights_path = Path('./weights_ub')/(writer_path.split('/')[1]+".pt")

# =====================    TRAINING  =======================

batch_size =16
validation_split = .2
dataset_size = len(pd.read_csv(all))
# train_indices, val_indices = utils_2d.generate_train_and_val_indices(dataset_size,validation_split=0.2,np_seed=False)

val_indices = np.load('val_indices.npy')
train_indices = np.load('train_indices.npy')

train= SegmentationDataset(all,root=root,indices=train_indices, include_t1ce=False,transform=rescale_only)
val= SegmentationDataset(all,root=root,indices=val_indices,include_t1ce=False, transform=rescale_only)


im, lab= train[0]
num_channels=im.shape[0]

im2, lab2 = train[1]
# Creating PT data samplers and loaders:
# train_sampler = SubsetRandomSampler(train_indices)
# valid_sampler = SubsetRandomSampler(val_indices)
#
#

loader_train = torch.utils.data.DataLoader(train, batch_size=batch_size)
loader_valid = torch.utils.data.DataLoader(val, batch_size=batch_size)

im,lab = next(iter(loader_train))
batch_shape = im.shape
#
# fig, ax = plt.subplots(1, 1)
# tracker = utils_2d.IndexTracker(ax,labl[0,:])
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
# plt.show()
# #
#
# fig3, ax3 = plt.subplots(1, 1)
# tracker3 = utils_2d.IndexTracker(ax3, lab[0,:])
# fig3.canvas.mpl_connect('scroll_event', tracker3.onscroll)
# plt.show()
#


parser = argparse.ArgumentParser(
    description="Training U-Net model for segmentation of brain MRI"
)


parser = argparse.ArgumentParser(
    description="Training U-Net model for segmentation of brain MRI"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=batch_size,
    help="input batch size for training (default: 16)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=100,
    help="number of epochs to train (default: 100)",
)
parser.add_argument(
    "--init_features",
    type=int,
    default=init_features,
)
parser.add_argument(
    "--num_channels",
    type=int,
    default=num_channels,
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    help="initial learning rate (default: 0.001)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="device for training (default: cuda:0)",
)
# parser.add_argument(
#     "--workers",
#     type=int,
#     default=4,
#     help="number of workers for data loading (default: 4)",
# )
parser.add_argument(
    "--vis-images",
    type=int,
    default=200,
    help="number of visualization images to save in log file (default: 200)",
)
parser.add_argument(
    "--vis-freq",
    type=int,
    default=10,
    help="frequency of saving images to log file (default: 10)",
)
parser.add_argument(
    "--weights", type=str, default=weights_path, help="folder to save weights"
)
parser.add_argument(
    "--logs", type=str, default="./logs", help="folder to save logs"
)
parser.add_argument(
    "--images", type=str, default="./kaggle_3m", help="root folder with images"
)
parser.add_argument(
    "--image-size",
    type=int,
    default=256,
    help="target input image size (default: 256)",
)
parser.add_argument(
    "--num_cases",
    default=[len(train),len(val)],
    help="List containing num cases in train and val datasets",
)
parser.add_argument(
    "--writer_path",
    default=writer_path
)


parser.add_argument(
    "--aug-scale",
    type=int,
    default=0.05,
    help="scale factor range for augmentation (default: 0.05)",
)
parser.add_argument(
    "--batch_shape",
    default=batch_shape,
    help="batch shape",
)
parser.add_argument(
    "--aug-angle",
    type=int,
    default=batch_shape,
    help="rotation angle range in degrees for augmentation (default: 15)",
)
args = parser.parse_known_args()[0]
# main_tr(args,loader_train,loader_valid)


#
#
#
#
# # =================     VIEW IMAGES=============================
#
#
# dat = SegmentationDataset(all,root=root,transform=composed)
# dat2 = SegmentationDataset(all,root=root)
# i=10
# a = dat[i]
# img = a[0]
# mask = a[1]
# im2 = dat2[i][0]
# mask2 = dat2[i][1]
# angle = 353
# plane = [3, 2]
# im3 = R(im2, axes=plane, angle=angle, reshape=False)
# r = Rotate3D('unet')
# im,lab = r(dat2[i])
# #
#
# fig, ax = plt.subplots(1, 1)
# tracker= utils_2d.IndexTracker(ax, im) # note slices are in the last index
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)


# fig3, ax3 = plt.subplots(1, 1)
# tracker3 = utils_2d.IndexTracker(ax3, im[0,:])
# fig3.canvas.mpl_connect('scroll_event', tracker3.onscroll)
# plt.show()
#
# fig4, ax4 = plt.subplots(1, 1)
# tracker4 = utils_2d.IndexTracker(ax4, lab[0,:])
# fig4.canvas.mpl_connect('scroll_event', tracker4.onscroll)
# plt.show()
#
#
#
#
#
#
#
#
#
#
#
# #================        view images ==========
# imgs = torch.load('mask_pred.pt')
# #
# img = img[1,0,:]
# print(torch.max(img))
# # tmp = img_as_bool(resize(mask.astype(bool), [256, 256, 256]))
# # tmp2 = img_as_bool(resize(mask,[256,256,256]))
# #
# #
# fig, ax = plt.subplots(1, 1)
# tracker = utils_2d.IndexTracker(ax,img)
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
# plt.show()
# #

#
# fig3, ax3 = plt.subplots(1, 1)
# tracker3 = utils_2d.IndexTracker(ax3, mask2)
# fig3.canvas.mpl_connect('scroll_event', tracker3.onscroll)
# plt.show()
#
# fig4, ax4 = plt.subplots(1, 1)
# tracker4 = utils_2d.IndexTracker(ax4,im2)
# fig4.canvas.mpl_connect('scroll_event', tracker4.onscroll)
# plt.show()
#
#
#
#
# #===========================================
# fl = a['flair_filename']
# fn = '/Users/ub/datasets/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_2013_0_1/BraTS19_2013_0_1_seg.npy'
#
# fl2 = np.load(fn)
# lims = utils_2d.bbox_3D(fl2)
# a= fl2[lims[0]:lims[1], lims[2]:lims[3], lims[4]:lims[5]]
#
# fl2 = dat[0][1]
# fig, ax = plt.subplots(1, 1)
# tracker = utils_2d.IndexTracker(ax, fl2)
#
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
#
#
#
# #1. to just get the names of csv files
# # lgg = get_files(lg, label='lgg')
# # hgg = get_files(hg)
# # lgg.to_csv(Path(main)/'LGG.csv', index =False)
# # hgg.to_csv(Path(main)/'HGG.csv', index =False)
#
# #2. run a loop to generate ni to numpy from files above
#
# #3. get numpy file listing using get_numpy_files(main)
#
#
# #4. process numpy files as dataset torch
#
# n, p = 1, .3  #  30% cases will be test set
#
# #Fusing HGG and LGG files, and generating random train/ test split
# # np_file='/Users/ub/datasets/MICCAI_BraTS_2019_Data_Training/LGG_numpy_dataset.csv'
# # lgg_df = pd.read_csv(np_file)
# # lgg_df['label'] = 'lgg'
# # len(lgg_df)
# #
# # hgg_df = pd.read_csv('/Users/ub/datasets/MICCAI_BraTS_2019_Data_Training/HGG_numpy_dataset.csv')
# # hgg_df['label'] = 'hgg'
# #
# # final= pd.concat(lgg_df,hgg_df)
# # final.to_csv(Path(main)/'allCases.csv',index=False)
#
# final_name = Path(main)/'allCases.csv'
#
# # composed = transforms.Compose([Rescale(64),ToTensor(),Normalize()])
# # composed = transforms.Compose([Rescale(64),ToTensor()])
# composed = transforms.Compose([Rescale(64),Rotate3D(), ToTensor(),Normalize()])
# composed = transforms.Compose([Rescale((16,128,128)),Rotate3D(), ToTensor(),Normalize()])
#
# dataset = pd.read_csv(final_name)
#
# batch_size =8
# validation_split = .2
# shuffle_dataset = True
# random_seed= 42
#
# dataset_size = len(dataset)
# indices = list(range(dataset_size))
# split = int(np.floor(validation_split * dataset_size))
# if shuffle_dataset :
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)
# train_indices, val_indices = indices[split:], indices[:split]
#
#
# train= TumourDataset(final_name,root=root,indices=train_indices, transform=composed)
# val= TumourDataset(final_name,root=root,indices=val_indices, transform=composed)
# weights_tr = train.class_weights
# weights_va = val.class_weights
# sampler_train= torch.utils.data.sampler.WeightedRandomSampler(weights_tr, len(weights_tr))
#
# sampler_val= torch.utils.data.sampler.WeightedRandomSampler(weights_va, len(weights_tr))
#
# # Creating PT data samplers and loaders:
# # train_sampler = SubsetRandomSampler(train_indices)
# # valid_sampler = SubsetRandomSampler(val_indices)
# #
# #
# train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
#                                            sampler=sampler_train)
# validation_loader = torch.utils.data.DataLoader(val, batch_size=batch_size,
#                                                 sampler=sampler_val)
#
#
#
# #
# # d = DataBunch(train_dl= train_loader, valid_dl= validation_loader)
# # a=d.one_batch()
# #
# # acc_02 = partial(accuracy_thresh, thresh=0.2)
# # f_score = partial(fbeta, thresh=0.2)
# # lr = 1e-2/2
# # net = ConvNet()
# # # l = Learner(d,net,metrics=[acc_02, f_score])
# # l = Learner(d,net,metrics=error_rate)
# #
# #
# # #=====================     VIEW IMAGE ========================
# df = pd.read_csv(all)
# row = df.iloc[34]
# a = np.load(Path(root)/row.t1ce_filename)
# b = np.expand_dims(a,0)
# c = np.load(Path(root)/row.flair_filename)
# d = np.expand_dims(c,0)
# # e =np.concatenate((b,d),axis=0)
# #
# b = (a-np.min(a))/(np.max(a)-np.min(a))
# c = torch.from_numpy(b)
# c=b.clone()
# d = b.clone()
# e = b.clone()
# for i in range(b.shape[0]):
#     c[i,:]=(c[i,:]-torch.min(c[i,:]))/(torch.max(c[i,:]-torch.min(c[i,:])))
#
#     b[i,:]=F.normalize(b[i,:],dim=(1,2))
#     print(torch.max(b[i,:]), torch.min(b[i,:]))
#
# c=F.normalize(b,dim=(0,1))
#
#
# f=c[100,:]
# F.normalize(c,dim=(0,1)).max()
#
# torch.max(b)
#
# fig2, ax2 = plt.subplots(1, 1)
# tracker2= utils_2d.IndexTracker(ax2, c)
# fig2.canvas.mpl_connect('scroll_event', tracker2.onscroll)
# #
# plt.show()
#
# # b = np.load(Path(root)/row.flair_filename)
# #
# msk = mask

#
#
# fig3, ax3 = plt.subplots(1, 1)
# tracker3= utils_2d.IndexTracker(ax3, mask[0,0,:])
# fig3.canvas.mpl_connect('scroll_event', tracker3.onscroll)
#



#
# #=============== ni t= numpy =============
#
# lg = pd.read_csv(main/'LGG.csv')
#
#
# id = 0
#
#
#
# for id in range(len(lg)):
#     row = lg.iloc[id]
#     ni_to_numpy(row,root,cropping='image')