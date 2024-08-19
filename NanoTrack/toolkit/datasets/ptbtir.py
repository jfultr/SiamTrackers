import json
import os

from tqdm import tqdm

from .dataset import Dataset
from .video import Video
import glob 

import numpy as np 


class PtbTirVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(PtbTirVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)


class PtbTirDataset(Dataset):
    """
    Args:
        name:  dataset name, should be ...
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(PtbTirDataset, self).__init__(name, dataset_root)

        self.anno_files = sorted(glob.glob(
            os.path.join(dataset_root, '*/*.txt')))
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]
        
        meta_data=dict()

        for  i in range(len(self.seq_names)):
            img_names = sorted(glob.glob(
                os.path.join(self.seq_dirs[i], 'img/*.jpg')))
            img_names=[x.split('/PTB-TIR/')[-1] for x in img_names]
            gt_rect = np.loadtxt(self.anno_files[i], delimiter=',')
            data=dict()
            data['video_dir']=self.seq_names[i]
            data['init_rect']=gt_rect[0]
            data['img_names']=img_names
            data['gt_rect']=gt_rect
            # data['attr']=0
            meta_data[self.seq_names[i]]=data


        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = PtbTirVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          None)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
