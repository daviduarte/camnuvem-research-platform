from torch.utils.data.dataloader import default_collate
import visdom
import numpy as np
from random import choice
import os
import pickle
from math import exp


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def make_gt():
    videos_pkl = "/home/lnn/workspace/pygcn/pygcn/ucf_crime_test.pkl"
    with open(videos_pkl, 'rb') as f:
        videos = pickle.load(f)
    src = "/home/lnn/workspace/UCF_Crimes/_iter_5000_c3d_ave_0/"
    dst = "/home/lnn/data/UCF_Crimes/test_pred_groundtruth/"
    for v in videos:
        src_file = os.path.join(src, v + "_c3d.npz")
        dst_file = os.path.join(dst, v + "_c3d.npz")
        f = np.load(src_file)
        begin_idx = f["begin_idx"].copy()
        scores = f["scores"].copy()
        gt = videos[v]
        ratio = float(len(gt)) / float(len(scores))
        for i in range(len(scores)):
            b = int(i * ratio + 0.5)
            e = int((i + 1) * ratio + 0.5)
            ans = np.mean(gt[b: e])
            ab_score = 1 if ans >= 0.5 else 0
            scores[i, :, 0] = ab_score
        np.savez(dst_file, scores=scores, begin_idx=begin_idx)


from sklearn.metrics import roc_auc_score

def evaluate_result(vid2abnormality):
    videos_pkl = "/home/lnn/workspace/pygcn/pygcn/ucf_crime_test.pkl"
    gt = []
    ans = []
    with open(videos_pkl, 'rb') as f:
        videos = pickle.load(f)
    for vid in videos:
        if not vid2abnormality.has_key(vid):
            print("The video %s is excluded on the result!" % vid)
            continue
        cur_ab = np.array(vid2abnormality[vid])
        cur_gt = np.array(videos[vid])
        ratio = float(len(cur_gt)) / float(len(cur_ab))
        cur_ans = np.zeros_like(cur_gt)
        for i in range(len(cur_ab)):
            b = int(i * ratio + 0.5)
            e = int((i + 1) * ratio + 0.5)
            cur_ans[b: e] = cur_ab[i]
        gt.extend(cur_gt.tolist())
        ans.extend(cur_ans.tolist())
    ret = roc_auc_score(gt, ans)
    print("Test AUC@ROC: %.4f" % ret)
    return ret



def process_feat(feat, length):
    new_feat = np.zeros((length, feat.shape[1])).astype(np.float32)
    
    r = np.linspace(0, len(feat), length+1, dtype=np.int)
    for i in range(length):
        if r[i]!=r[i+1]:
            new_feat[i,:] = np.mean(feat[r[i]:r[i+1],:], 0)
        else:
            new_feat[i,:] = feat[r[i],:]
    return new_feat


class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_lines(self, name, y, **kwargs):
        '''
        self.plot('loss', 1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=str(name),
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1
    def disp_image(self, name, img):
        self.vis.image(img=img, win=name, opts=dict(title=name))
    def lines(self, name, line, X=None):
        if X is None:
            self.vis.line(Y=line, win=name)
        else:
            self.vis.line(X=X, Y=line, win=name)
    def scatter(self, name, data):
        self.vis.scatter(X=data, win=name)    

def save_best_record(auc, auc2, epoch, file_path):
    fo = open(file_path, "w")
    fo.write("epoch: {}\n".format(epoch))
    fo.write(str(auc))
    fo.write("\n"+str(auc2))
    fo.close()        