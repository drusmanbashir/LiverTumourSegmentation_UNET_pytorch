import platform, os, argparse
# import matplotlib
# matplotlib.use('Qt5agg')
# os.environ['KMP_DUPyyLICATE_LIB_OK']='True'# to avoid sigseg
# import matplotlib.pyplot as plt
from prep_data_LITS import LITS_dataset, getDatasize, ni_to_tensor_CT_folder
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from transforms_ub import RandomFlipHorizontal

# from train_ub import main as main_tr
#


from torchvision import transforms
if platform.system()=='Darwin':  # macbook vs gcp
    import importlib.util
    root = '/Users/ub'
    specs = importlib.util.spec_from_file_location("utils", "/Users/ub/Dropbox/code/utils/utils_2d.py")
    utils_2d= importlib.util.module_from_spec(specs)
    specs.loader.exec_module(utils_2d)
else:
    root = '/home/ub'
    import utils_2d
    nii_fldr = Path(root) /'datasets/LITS/TrainingAll'

tensors_fldr = Path(root)/'datasets/LITS/tensors'
# ni_to_tensor_CT_folder(foldr=nii_fldr,overwrite=False)



#============arguments ======================================
batch_size =1
validation_split = .2
dataset_size = getDatasize(tensors_fldr)
lr = 0.00001
init_features =16
writer_path='runs/unet_LITS_16b'  #location to save log files in
save_weights_path = Path('./weights_ub')/(writer_path.split('/')[1]+".pt")
load_weights_path =Path('./weights_ub')/''

i=0
num_classes = i+1
modes =['metastases','liver','liver_and_metastases']
mode = modes[i]
proportion_processed= 0.6   #only process 60% cases in both training and validation

#=========== prep data ======================
train_indices, val_indices = utils_2d.generate_train_and_val_indices(dataset_size,np_seed=False, validation_split=0.2)

composed = transforms.Compose([RandomFlipHorizontal()])

train=LITS_dataset(tensors_fldr,indices=train_indices,mode=mode,transform=None)
val  = LITS_dataset(tensors_fldr,mode=mode,indices=val_indices)

loader_train = torch.utils.data.DataLoader(train, batch_size=batch_size)
loader_valid = torch.utils.data.DataLoader(val, batch_size=batch_size)

iteri = iter(loader_train)

im,lab = next((iteri))
batch_shape = im.shape
print(torch.max(lab[:]))





parser = argparse.ArgumentParser(
    description="Training U-Net model for segmentation of liver ct"
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=batch_size,
    help="input batch size for training (default: 16)",
)



parser.add_argument(
    "--num_classes",
    type=int,
    default=num_classes,
    help="input batch size for training (default: 16)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=25,
    help="number of epochs to train (default: 100)",
)
parser.add_argument(
    "--init_features",
    type=int,
    default=init_features,
)

parser.add_argument(
    "--lr",
    type=float,
    default=lr,
    help="initial learning rate (default: 0.001)",
)
parser.add_argument(
    "--proportion_processed",
    type=float,
    default=proportion_processed,
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda:0",
    help="device for training (default: cuda:0)",
)

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
    "--load_weights", type=str, default=load_weights_path, help="folder to save weights"
)
parser.add_argument(
    "--save_weights", type=str, default=save_weights_path, help="folder to save weights"
)
parser.add_argument(
    "--logs", type=str, default="./logs", help="folder to save logs"
)
parser.add_argument(
    "--writer_path",
    default=writer_path
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

args = parser.parse_known_args()[0]

