from dateutil.relativedelta import relativedelta
from src.model_utils import get_model
from src.utils import pad_collate

from argparse import Namespace
from datetime import datetime
from glob import glob

import pandas as pd
import numpy as np

import torch
import json
import sys
import os

from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def load_model(path, device, fold=1, mode='semantic'):
    """Load pre-trained model"""
    with open(os.path.join(path, 'conf.json')) as file:
        config = json.loads(file.read())
    config = Namespace(**config)
    model = get_model(config, mode = mode).to(device)

    sd = torch.load(
        os.path.join(path, "Fold_{}".format(fold+1), "model.pth.tar"),
        map_location=device
        )
    model.load_state_dict(sd['state_dict'])
    return model


def get_cloud_coverage(x, thresh=5):
    img = np.load(x).sum(axis=0)
    return (img > thresh).sum() / np.product(img.shape)


def get_data_df(PATH_TO_DATA, fold=3, dateref=datetime(2022, 9, 1), fovThresh = 0.2):
    flist = glob(PATH_TO_DATA+'20*.npy')

    df = pd.DataFrame(data=flist, columns=['fname'])

    df['date'] = df.fname.apply(lambda x: datetime.strptime(x.split('/')[-1].split('.')[0], '%Y%m%d'))
    df['deltad'] = df.date.apply(lambda x: (x - dateref).days)
    df['fmasks'] = df['fname'].apply(lambda x: x[:-12] + 'cloudProb/' + x[-12:])

    df = df.set_index('date').sort_index()
    ## Read Mask Data
    return df


def get_norm_factor(PATH_TO_NORM, fold=3):
    ## Get Normalization Factor
    with open(os.path.join(PATH_TO_NORM, "NORM_S2_patch.json"), "r") as file:
        normvals = json.loads(file.read())
    means = np.array(normvals["Fold_{}".format(fold)]["mean"])
    stds  = np.array(normvals["Fold_{}".format(fold)]["std"])
    return means, stds


def get_data(df, PATH_TO_NORM, nbands=10):
    means, stds = get_norm_factor(PATH_TO_NORM)

    data = np.array([np.load(fname) for fname in df.fname]).astype(float)
    for i in range(10): data[:, i] = (data[:, i] - means[i]) / stds[i]
    x = torch.from_numpy(data).double()
    x = x.type(torch.FloatTensor).to(device).unsqueeze(0)
    dates = torch.from_numpy(df['deltad'].values)
    dates = dates.to(device).unsqueeze(0)
    return x, dates


def main(PATH_TO_REPO, PATH_TO_DATA, PATH_TO_NORM, PATH_TO_UTAE_WEIGHTS,
         PATH_TO_UTAEPaPs_WEIGHTS, device, utae, fovCloudThresh=0.05):

    ## Reading Events Data
    df = get_data_df(PATH_TO_DATA)

    ## Get Cloud Coverage Percentage -> Filtering invalid data
    df['cloudPerc'] = df['fmasks'].apply(get_cloud_coverage)
    df = df[df.cloudPerc < fovCloudThresh]
    ## Computing Predictions
    x, dates = get_data(df, PATH_TO_NORM)

    with torch.no_grad():
        sempred = utae(x, batch_positions=dates)
        #sempred = sempred.argmax(dim=1)
    
    sempred = sempred.max(dim=1)
    sempred = np.concatenate([  sempred.indices.detach().cpu().numpy(), 
                                sempred.values.detach().cpu().numpy()],
                                axis=0) 
    np.save(PATH_TO_DATA+'/crops_segmentation.npy', sempred)


if __name__ == "__main__":
    from os.path import join

    root_dir= '/projects/DeepLeey/agri-tech/sentinel-segmentation/crop-type/sentinel-crops-dectection/'

    PATH_TO_REPO             = join(root_dir, 'utae-paps/')
    PATH_TO_DATA             = '/data/2/Sentinel2-crops/*/*/'
    PATH_TO_NORM             = join(root_dir, 'utae-paps')
    PATH_TO_UTAE_WEIGHTS     = join(root_dir, 'UTAE_zenodo')
    PATH_TO_UTAEPaPs_WEIGHTS = join(root_dir, 'UTAE_PAPs')

    device                   = torch.device('cuda:0')

    sys.path.append(PATH_TO_REPO)

    utae = load_model(PATH_TO_UTAE_WEIGHTS, device=device, fold=3,
                      mode='semantic').eval()

    for x in tqdm(glob(PATH_TO_DATA)):
        main(PATH_TO_REPO, x, PATH_TO_NORM, PATH_TO_UTAE_WEIGHTS,
             PATH_TO_UTAEPaPs_WEIGHTS, device, utae)
