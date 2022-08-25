import argparse
parser = argparse.ArgumentParser(description='RTFM')

parser.add_argument('--gpu-id', help="Gpu ID: e.g. cuda:0", default="cuda:1")

parser.add_argument('--replace', help="Replace files already created? If not, feature extraction will occur only when the .npy files not exist.", default="False")

parser.add_argument('--feature-extractor', help="What architecture use to extract features? i3d / c3d", default="False")

parser.add_argument('--root', help="CamNuvem root folder", default="/media/denis/526E10CC6E10AAAD/CamNuvem")

parser.add_argument('--re-run-test', help="Re-run the 10-test in a checkpoint", default="False")
parser.add_argument('--checkpoint', help="Load a checkpoint", default="False")

parser.add_argument('--no-anomaly-detection', help="Only feature extraction. Does not perform anomaly detection training", default="False")
parser.add_argument('--no-feature-extraction', help="Use features already extracted. perform only the anomaly training", default="False")

parser.add_argument('--crop-10', help="Load a checkpoint", default="True")

parser.add_argument('--video-root-test-abnormal', help="Complete path for mp4 test abnormal videos", default="False")
parser.add_argument('--video-root-train-abnormal', help="Complete path for mp4 train abnormal videos", default="False")
parser.add_argument('--video-root-test-normal', help="Complete path for mp4 test normal videos", default="False")
parser.add_argument('--video-root-train-normal', help="Complete path for mp4 train normal videos", default="False")
#parser.add_argument('--video-path', help="Path the videos are saved")

parser.add_argument('--feature-root-test-abnormal', help="Path where the resulting test abnormal .npy files will be saved", default="False")
parser.add_argument('--feature-root-train-abnormal', help="Path where the resulting test abnormal .npy files will be saved", default="False")
parser.add_argument('--feature-root-test-normal', help="Path where the resulting test abnormal .npy files will be saved", default="False")
parser.add_argument('--feature-root-train-normal', help="Path where the resulting test abnormal .npy files will be saved", default="False")

parser.add_argument('--segment-size', type=int, help="How much frames to use to extract the feature vector?", default=16)


# For create list files for RTFM
parser.add_argument('--make-list-file', help="Make the file .list with the path of abnormal and normal train and test files.", default="False")
parser.add_argument('--test-list-file-final-name', help="What is the complete path + name of the resulting test list file?", default="False")
parser.add_argument('--training-list-file-final-name', help="What is the complete path + name of the resulting training list file?", default="False")
#parser.add_argument('--i3d-root-train-abnormal', help="Path for .npy files with the ABNORMAL TRAIN's i3d vector", default="False")
#parser.add_argument('--i3d-root-train-normal', help="Path for .npy files with the NORMAL TRAIN's i3d vector", default="False")
#parser.add_argument('--i3d-root-test-abnormal', help="Path for .npy files with the ABNORMAL TEST's i3d vector", default="False")
#parser.add_argument('--i3d-root-test-normal', help="Path for .npy files with the NORMAL TEST's i3d vector", default="False")

# For WSAL anomaly detection
parser.add_argument('--make-hd5-file', help="Make Hd5 files?", default="False")

# Create ground truth
parser.add_argument('--make-gt', help="Make ground truth file for test videos.", default="False")
parser.add_argument('--test-labels', help="The file containing all frame-level anomaly in test videos.", default="False")

parser.add_argument('--gt-only-anomaly', default='False', help='file of ground truth ')

# Args from RTFM
#parser = argparse.ArgumentParser(description='RTFM')
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'])
parser.add_argument('--feature-size', type=int, default=2048, help='size of feature (default: 2048)')
parser.add_argument('--modality', default='RGB', help='the type of the input, RGB,AUDIO, or MIX')
parser.add_argument('--rgb-list', default='list/camnuvem-i3d-train-10crop.list', help='list of rgb features ')
parser.add_argument('--test-rgb-list', default='list/camnuvem-i3d-test-10crop.list', help='list of test rgb features ')
parser.add_argument('--gt', default='False', help='file of ground truth ')
parser.add_argument('--gpus', default=1, type=int, choices=[0], help='gpus')
parser.add_argument('--lr', type=str, default='[0.001]*15000', help='learning rates for steps(list form)')
parser.add_argument('--batch-size', type=int, default=16, help='number of instances in a batch of data (default: 16)')
parser.add_argument('--workers', default=8, help='number of workers in dataloader')
parser.add_argument('--model-name', default='rtfm', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--dataset', default='camnuvem', help='dataset to train on (default: )')
parser.add_argument('--plot-freq', type=int, default=10, help='frequency of plotting (default: 10)')
parser.add_argument('--max-epoch', type=int, default=5000, help='maximum iteration to train (default: 500)')

