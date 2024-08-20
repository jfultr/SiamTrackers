import json
import os

from tqdm import tqdm
from PIL import Image

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
        
        self.gt_traj = [[0] if np.isnan(bbox[0]) else bbox
                for bbox in self.gt_traj]  

        self.confidence = {}
    
    def load_tracker(self, path, tracker_names=None, store=True):
            """
            Args:
                path(str): path to result
                tracker_name(list): name of tracker
            """
            if not tracker_names:
                tracker_names = [x.split('/')[-1] for x in glob(path)
                        if os.path.isdir(x)]
            if isinstance(tracker_names, str):
                tracker_names = [tracker_names]
            for name in tracker_names:
                traj_file = os.path.join(path, name, 'longterm',
                        self.name, self.name+'_001.txt')
                with open(traj_file, 'r') as f:
                    traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                if store:
                    self.pred_trajs[name] = traj
                confidence_file = os.path.join(path, name, 'longterm',
                        self.name, self.name+'_001_confidence.value')
                with open(confidence_file, 'r') as f:
                    score = [float(x.strip()) for x in f.readlines()[1:]]
                    score.insert(0, float('nan'))
                if store:
                    self.confidence[name] = score
            return traj, score


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
