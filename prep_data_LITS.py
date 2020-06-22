from pathlib import Path
import os,platform
import importlib
from transforms_ub import normalize_numpy
import torch,re
import nibabel as nib
import numpy as np
import pkbar
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use('Tkagg')
import torch.nn.functional as F

if platform.system()=='Darwin':  # macbook vs gcp
    root = '/Users/ub'
    spec = importlib.util.spec_from_file_location("utils", "/Users/ub/Dropbox/code/utils/utils_2d.py")
    utils_2d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_2d)
else:
    root = '/home/ub'
    # import utils_2d



def ni_to_tensor_CT_folder(foldr,as_patches=False,overwrite=False):
   #
    if as_patches:
        patches_fldr = foldr.parents[4]/"patches"
        if not patches_fldr:
            os.makedirs(patches_fldr)
    else:
        tensor_fldr = foldr.parent/"tensors"
        if not tensor_fldr.exists():
            os.makedirs(tensor_fldr)

    directory = os.listdir(foldr)
    vols = [f for f in directory if 'volume' in f]
    bar = pkbar.Kbar(target = len(vols),width=28)
    for idx in range(len(vols)):
        bar.update(idx)#,values =[("Processing case:" ,vol_name)])
        vol_name = vols[idx]
        mask_name = vol_name.replace('volume','segmentation')

        if not as_patches:
            vol_outname = tensor_fldr/vol_name.replace(".nii",  ".pt")  # 4 quadrant given indexed suffixes and saved
            mask_outname =tensor_fldr/mask_name.replace(".nii",".pt")

            if overwrite or not vol_outname.exists():
                img_np = np.array(nib.load(Path(foldr) / vol_name).dataobj)
                vol_norm_pt = torch.from_numpy(normalize_numpy(img_np)).float().unsqueeze(0)  # read array ->normalize ->torch float tensor -> add channel dimension
                torch.save(vol_norm_pt, vol_outname)

            if overwrite or not mask_outname.exists():
                mask_np = np.array(nib.load(Path(foldr) / mask_name).dataobj)
                mask_pt = torch.from_numpy(mask_np.astype(float)).unsqueeze(0).float()
                torch.save(mask_pt, mask_outname)
        else:  #save as patches -- very large dataset so this code below is untested
            img_np = np.array(nib.load(Path(foldr) / vol_name).dataobj)
            vol_norm_pt = torch.from_numpy(normalize_numpy(img_np)).float().unsqueeze(0)  # read array ->normalize ->torch float tensor -> add channel dimension
            img=[vol_norm_pt[:,0:256,0:256,:],vol_norm_pt[0,256:512,0:256,:],vol_norm_pt[:,0:256,256:512],vol_norm_pt[:,256:512,256:512]]
            mask_np = np.array(nib.load(Path(foldr) / mask_name).dataobj)
            mask_pt = torch.from_numpy(mask_np.astype(float)).unsqueeze(0).float()
            mask=[mask_pt[:,0:256,0:256,:],mask_pt[:,256:512,0:256,:],mask_pt[:,0:256,256:512],mask_pt[:,256:512,256:512]]

            for i in range(4):
             vol_outname =vol_name.replace(".nii","_patch"+str(i)+".pt")  # 4 quadrant given indexed suffixes and saved
             vol_outname = patches_fldr / str(vol_outname)
             if overwrite or not vol_outname.exists():
                 torch.save(img[i],vol_outname)
             mask_outname = mask_name.replace(".nii", "_patch"+str(i)+".pt")
             mask_outname =patches_fldr/str(mask_outname)
             if overwrite or not mask_outname.exists():
                torch.save(mask[i],mask_outname)


def  verify_dataintegrity(tensors_fldr,vol_mask_pairs):
    print("Verifying dataset volumes / mask pairs are correct and exist..")
    for idx in range(len(vol_mask_pairs)):
        vol = vol_mask_pairs[idx][0]
        mask = vol_mask_pairs[idx][1]
        assert (vol.replace("volume","segmentation")==mask and (tensors_fldr/vol).exists() and (tensors_fldr/mask).exists()),"Check pair "+vol
    print("Done.. Data clean")

def getDatasize(tensors_fldr):
    directory = os.listdir(tensors_fldr)
    vols = [Path(tensors_fldr) / f for f in directory if 'volume' in f]
    return (len(vols))  #multiply by 4 because each case has 4 patches

class LITS_dataset(torch.utils.data.Dataset):# total 131 cases
    def __init__(self,tensors_fldr,mode, indices=[], transform=None): #outerzone of tumour mask i.e., values 2 , 3, or 4
        assert mode in ["metastases","liver","liver_and_metastases"], "Mode should be either 'metastases', 'liver, or 'liver_and_metastases' for 1 class or 2class model"
        directory = os.listdir(tensors_fldr)
        self.tensors_fldr=tensors_fldr
        self.mode = mode
        vols = [f for f in directory if 'volume' in f]
        masks = [f.replace("volume","segmentation") for f in vols]
        self.vol_mask_pairs = list(zip(vols,masks))
        verify_dataintegrity(self.tensors_fldr,self.vol_mask_pairs)
        self.transform = transform
        self.indices=indices
        if (len(indices)>0):
            self.vol_mask_pairs=[self.vol_mask_pairs[indx] for indx in indices]
    def __len__(self):
        return (len(self.vol_mask_pairs))  #each item has 4 patches

    def __getitem__(self, idx):

        volume_fn = self.vol_mask_pairs[idx][0]
        mask_fn = self.vol_mask_pairs[idx][1]

        volume = torch.load(self.tensors_fldr/volume_fn)
        mask = torch.load(self.tensors_fldr/mask_fn)

        if self.mode =='metastases':  # 'all pixels labelled 2 ie lesions are labeled 1 and everyting else (including liver) is labeled b/g ie 0
            mask[mask<2]=0
            mask[mask==2]=1
        if self.mode =='liver':  # 'all pixels labelled 2 ie lesions are labeled 1 and everyting else (including liver) is labeled b/g ie 0
            mask[mask>1]=1
        sample = [volume, mask]
        if self.transform:
            sample = self.transform(sample)
        return sample


#loading nii then converting to tensors is slower than loading tensors directly
class LITS_dataset_nii(torch.utils.data.Dataset):# total 131 cases
    def __init__(self,tensors_fldr,mode, indices=[], transform=None): #outerzone of tumour mask i.e., values 2 , 3, or 4
        assert mode in ["metastases","liver_and_metastases"], "Mode should be either 'metastases' or 'liver_and_metastases' for 1 class or 2class model"
        self.mode = mode
        self.tensors_fldr=tensors_fldr
        directory = os.listdir(tensors_fldr)
        vols = [f for f in directory if 'volume' in f]
        masks = [f.replace("volume","segmentation") for f in vols]
        self.vol_mask_pairs = list(zip(vols,masks))
        verify_dataintegrity(tensors_fldr,self.vol_mask_pairs)
        self.transform = transform
        self.indices=indices
        if (len(indices)>0):
            self.vol_mask_pairs=[self.vol_mask_pairs[indx] for indx in indices]
    def __len__(self):
        return (len(self.vol_mask_pairs)*4)  #each item has 4 patches

    def __getitem__(self, idx):
        index,subindex = divmod(idx,4)
        lims = [0,256,512]
        f,s= divmod(subindex,2)
        volume_fn = self.vol_mask_pairs[index][0]
        mask_fn = self.vol_mask_pairs[index][1]
        img_np = np.array(nib.load(Path(self.tensors_fldr) / volume_fn).dataobj)
        volume= torch.from_numpy(normalize_numpy(img_np)).float().unsqueeze(0)  # read array ->normalize ->torch float tensor -> add channel dimension
        mask_np = np.array(nib.load(Path(self.tensors_fldr) / mask_fn).dataobj)
        mask = torch.from_numpy(mask_np.astype(float)).unsqueeze(0).float()
        subVol= volume[:,lims[f]:lims[f+1],lims[s]:lims[s+1],:]
        subMask =mask[:,lims[f]:lims[f+1],lims[s]:lims[s+1],:]
        if self.mode =='metastases':  # 'all pixels labelled 2 ie lesions are labeled 1 and everyting else (including liver) is labeled b/g ie 0
            subMask[subMask<2]=0
            subMask[subMask==2]=1
        sample = [subVol, subMask]
        if self.transform:
            sample = self.transform(sample)
        return sample




def get_bounding_boxes(root,fldr='datasets/LITS/nas/01_Datasets/CT/LITS/Trainingall'):   #DF index column is caseID
 directory = os.listdir(Path(root)/fldr)
 masks = [f for f in directory if 'segmentation' in f]
 header = ["no_slices","left","right","posterior","anterior","bottom","top","slice_thickness","voxel_width","bbox_width",'bbox_ap','bbox_height']
 df = pd.DataFrame(columns=header)
 # for idx in range(0,10):
 for idx in range(len(masks)):
    fname =  masks[idx]
    pat = r"\d{1,3}\b"
    caseid = re.findall(pat, fname)
    nii_file = nib.load(Path(root)/fldr / fname)
    ni_header= nii_file.header
    slice_thickness = ni_header['pixdim'][3]
    voxel_width=ni_header['pixdim'][1]
    mask_np = np.array(nii_file.dataobj)
    num_slices = mask_np.shape[2]
    lims = utils_2d.bbox_3D(mask_np)
    id=int(caseid[0])
    listi = [num_slices,lims]
    df.loc[id]=({header[0]:listi[0],header[1]:listi[1][0],header[2]:listi[1][1],header[3]:listi[1][2],header[4]:listi[1][3],header[5]:listi[1][4],\
                 header[6]:listi[1][5],header[7]:slice_thickness,header[8]:voxel_width,header[9]:(lims[1]-lims[0]),header[10]:(lims[3]-lims[2]),header[11]:(lims[5]-lims[4])})
 return df.sort_index()


#


def gen_limited_images_frombbox(nii_fldr,out_fldr,bbox_df,overwrite=False,outsize=256):
        for idx in range(len(bbox_df)):
            print("Processing case id: ", idx)
            row = bbox_df.loc[idx].astype(int)
            slice_thickness=row.slice_thickness
            # row = bbox_df.iloc[idx]
            vol_name =Path(nii_fldr)/('volume-'+str(idx)+'.nii')
            vol_outname = Path(out_fldr)/('volume-'+str(idx)+'.pt')
            mask_name=Path(nii_fldr)/('segmentation-'+str(idx)+'.nii')
            mask_outname = Path(out_fldr) / ('segmentation-' + str(idx) + '.pt')

            if not vol_outname.exists()or overwrite==True:
                img_np=np.array(nib.load(vol_name).dataobj)
                extra_top_slice = np.floor_divide(row.bbox_height%4,2)
                extra_bot_slice = int(np.ceil((row.bbox_height%4)/2))
                img_cropped= img_np[row.left_final:row.right_final,row.posterior_final:row.anterior_final,(row.bottom-extra_bot_slice):(row.top+extra_top_slice)]
                num_slices = int(img_cropped.shape[2]/2) if slice_thickness<2 else img_cropped.shape[2]  #if v thin slices then interpolate in z direction as well
                img_norm= normalize_numpy(img_cropped)
                img_pt = torch.tensor(img_norm).float().unsqueeze_(0).unsqueeze_(0)#one dim for channel one for batch (to allow F.resize)
                img_pt_interp=F.interpolate(img_pt, (outsize, outsize,num_slices))
                torch.save(img_pt_interp.squeeze(0),vol_outname)

            if not mask_outname.exists()or overwrite==True:
                mask_np=np.array(nib.load(mask_name).dataobj)
                mask_cropped = mask_np[row.left_final:row.right_final,row.posterior_final:row.anterior_final,(row.bottom-extra_bot_slice):(row.top+extra_top_slice)]
                num_slices = int(mask_cropped.shape[2]/2) if slice_thickness<2 else mask_cropped.shape[2]  #if v thin slices then interpolate in z direction as well
                mask_pt= torch.tensor(mask_cropped.astype(int)).float().unsqueeze_(0).unsqueeze_(0)#one dim for channel one for batch (to allow F.resize)
                mask_pt_interp =F.interpolate(mask_pt, (outsize, outsize,num_slices))
                torch.save(mask_pt_interp.squeeze(0),mask_outname)


from torchvision import transforms
import importlib.util
if platform.system()=='Darwin':  # macbook vs gcp
    root = '/Users/ub'
    specs = importlib.util.spec_from_file_location("utils", "/Users/ub/Dropbox/code/utils/utils_2d.py")
    utils_2d= importlib.util.module_from_spec(specs)
    specs.loader.exec_module(utils_2d)
    nii_fldr = Path(root) /'datasets/LITS/nas/01_Datasets/CT/LITS/Trainingall'

else:
    root = '/home/ub'
    nii_fldr = Path(root) /'datasets/LITS/TrainingAll'
    import utils_2d

tensors_fldr= Path(root) /'datasets/LITS/tensors'
cropped_fldr = Path(root)/'datasets/LITS/cropped_liveronly'


def main():
    # d = get_bounding_boxes(root=root)
    # d.to_csv("boundingboxes.csv")
    bbox_df = pd.read_csv("boundingboxes_excel_modified.csv", index_col=0)
    gen_limited_images_frombbox(nii_fldr,tensors_fldr,overwrite=True,bbox_df=bbox_df)

    fig, ax = plt.subplots(1, 1)
    tracker = utils_2d.IndexTracker(ax,img_cropped)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    #


    fig3, ax3 = plt.subplots(1, 1)
    tracker3 = utils_2d.IndexTracker(ax3, img_pt_interp[0,0,:])
    fig3.canvas.mpl_connect('scroll_event', tracker3.onscroll)
    plt.show()

    fig4, ax4 = plt.subplots(1, 1)
    tracker4 = utils_2d.IndexTracker(ax4,mask_pt_interp[0,0,:])
    fig4.canvas.mpl_connect('scroll_event', tracker4.onscroll)
    plt.show()
