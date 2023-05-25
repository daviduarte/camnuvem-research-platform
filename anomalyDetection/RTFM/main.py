import sys
#sys.path.append("/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/RTFM")
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from .utils import save_best_record
from .model import Model
from .dataset import Dataset
from .train import train as trainRTFM
from .test_10crop import test
from . import rtfm_option
from tqdm import tqdm
from .utils import Visualizer
from .config import *
import time


viz = Visualizer(env='RTFM', use_incoming_socket=False)

def train(args):

    config = Config(args) 

    print("Carregando o train n_loader")
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=0, pin_memory=False, drop_last=True)

    print("Carregando o train a_loader")
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=False,
                               num_workers=0, pin_memory=False, drop_last=True)

    print("Carregando o test_loader")
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)

    test_loader_only_anomaly = DataLoader(Dataset(args, test_mode=True, only_anomaly=True),        # Test mode with only anomaly videos
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)    

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(args.gpu_id)
    model = Model(args.feature_size, args.batch_size, args.crop_10, device)
    model = model.to(device)

    if args.re_run_test == "True":

        if args.checkpoint == "False":
            print("args.checkpoint has to be setted")
            exit()
        
        print("Carregando o checkpoint")
        print(args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint, map_location='cuda:0'))

        # Test
        auc1 = test(test_loader, model, args, viz, device, args.gt)

        # Test now with only the anomaly videos
        auc2 = test(test_loader_only_anomaly, model, args, viz, device, args.gt_only_anomaly, only_abnormal=True)

        print("auc1: ")
        print(auc1)
        print(auc2)

        exit()

    for name, value in model.named_parameters():
        print(name)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    print(config.lr[0])
    time.sleep(1)

    optimizer = optim.Adam(model.parameters(),
                            lr=0.001, eps=1e-3, weight_decay=0.005)

    test_info = {"epoch": [], "test_AUC": []}
    best_AUC = -1
    output_path = './anomalyDetection/RTFM/ckpt'   # put your own path here
    print("Entrando no test")
    #time.sleep(3)
    auc = test(test_loader, model, args, viz, device, args.gt)

    contzera = 0
    for step in tqdm(
            range(1, args.max_epoch + 1),
            total=args.max_epoch,
            dynamic_ncols=True
    ):
        if step > 1 and config.lr[step - 1] != config.lr[step - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[step - 1]

        #print(len(train_nloader))
        #exit()
        if (step - 1) % len(train_nloader) == 0:
            loadern_iter = iter(train_nloader)

        if (step - 1) % len(train_aloader) == 0:
            loadera_iter = iter(train_aloader)

        trainRTFM(loadern_iter, loadera_iter, model, args.batch_size, optimizer, viz, device, contzera)
        contzera += 1
        if step % 50 == 0 and step > 10:

            auc = test(test_loader, model, args, viz, device, args.gt)
            test_info["epoch"].append(step)
            test_info["test_AUC"].append(auc)

            if test_info["test_AUC"][-1] > best_AUC:
                best_AUC = test_info["test_AUC"][-1]
                torch.save(model.state_dict(), os.path.join(output_path, args.model_name + '{}-i3d.pkl'.format(step)))
                save_best_record(test_info, os.path.join(output_path, '{}-step-AUC.txt'.format(step)))
    torch.save(model.state_dict(), os.path.join(output_path, args.model_name + 'final.pkl'))

if __name__ == '__main__':

    args = rtfm_option.parser.parse_args()   
    train(args)
