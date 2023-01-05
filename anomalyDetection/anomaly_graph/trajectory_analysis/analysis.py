import numpy as np
import torch


def calculeObjectPath(graph, frame, obj_index):

    object_path = []

    for f in graph[frame:]:     # For each frame
        obj = f[obj_index]

        if obj == -1:
            return  object_path

        object_path.append(obj)
        
    return object_path


def calculeFrameSimilarity(start_frame, path, score_list, bbox_fea_list, box_list, KEY_FRAME_SIM, DEVICE):

    nem_frames = len(bbox_fea_list) + 1

    vet1 = []
    vet2 = []

    for i in range(len(path)-1):
        i = i+1 # Key frame is compared with the predecessor

        # Get the current fea frame        
        if start_frame + i < nem_frames-1:
            fea = bbox_fea_list[i+start_frame][0][path[i]]
        else:
            fea = bbox_fea_list[i+start_frame-1][1][path[i]]    
        
        # get the predecessor fea frame
        i = i-1
        if start_frame + i < nem_frames-1:
            fea_pre = bbox_fea_list[i+start_frame][0][path[i]]
        else:
            fea_pre = bbox_fea_list[i+start_frame-1][1][path[i]]    

        #fea = torch.from_numpy(fea).to(DEVICE)
        #fea_pre = torch.from_numpy(fea_pre).to(DEVICE)

        vet1.append(fea)
        vet2.append(fea_pre)


    vet1 = np.stack(vet1, axis=0)
    vet2 = np.stack(vet2, axis=0)
    vet1 = torch.from_numpy(vet1).to(DEVICE)
    vet2 = torch.from_numpy(vet2).to(DEVICE)

    
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08).to(DEVICE)
    sim = cos_sim(vet1, vet2)
    

    # If sim < theshold, we consider the keyframe has changed
    key_frames = sim < KEY_FRAME_SIM
    key_frames[0] = True # We have to have at 1 keyframe, i.e., the first
    position = torch.where(key_frames==True, )    # Remember where each key_frame is
    key_frames = vet2[key_frames]



    return key_frames, position[0]
