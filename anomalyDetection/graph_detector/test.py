import torch
import util.utils as utils
import temporalGraph
import numpy as np
from definitions import FRAMES_DIR
import os

def test(model, loss, test_loader, reference_frame, obj_predicted, viz, buffer_size, DEVICE, EXIT_TOKEN, N, SIMILARITY_THRESHOLD, T, OBJECTS_ALLOWED, STRIDE):    
    print("Testing")
    temporal_graph = temporalGraph.TemporalGraph(DEVICE, buffer_size, OBJECTS_ALLOWED, N, STRIDE)
    data_loader_test = iter(test_loader)
    print("O dataset de teste tem: " + str(len(data_loader_test)) + " amostras")
    with torch.no_grad():
        model.eval()

        loss_mean = 0
        for i in range(len(data_loader_test)):
            print(i)
            input = next(data_loader_test)
            folder_index = input[1][0]
            sample_index = input[2][0]
            input = input[0]            
            input = np.squeeze(input)


            cache_folder = "cache_pt_task/test/T="+str(T)+"-N="+str(N)+"/"
            data_path = os.path.join(FRAMES_DIR, cache_folder, str(folder_index.cpu().numpy()), str(sample_index.cpu().numpy())+"_data.npy")
            print(data_path)
            has_cache = False
            if os.path.exists(data_path):
                has_cache = True
                test_loader.has_cache = True
            else:
                has_cache = False
                
            if not has_cache:
                adj_mat, bbox_fea_list, box_list, score_list = temporal_graph.frames2temporalGraph(input, folder_index, sample_index)

                SIMILARITY_THRESHOLD = 0.65#0.73
                graph = utils.calculeTargetAll(adj_mat, bbox_fea_list, box_list, score_list, reference_frame, SIMILARITY_THRESHOLD, T, N)


                # If in the first frame there is no object detected, so we have nothing to do here
                # The number of detected objects may be less than N. In this case we have nothing to do here
                #if len(bbox_fea_list[reference_frame][obj_predicted]) < N:
                #    print("continuando")
                #    continue       # Continue

                data, object_path = utils.calculeTarget(graph, score_list, bbox_fea_list, box_list, reference_frame, obj_predicted, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)

                path = os.path.join(FRAMES_DIR, cache_folder, str(folder_index.cpu().numpy()))
                os.makedirs(path, exist_ok=True)
                np.save(data_path, data)
            else:                
                print("Ok, temos cache, vamos carregar")
                data = np.load(data_path, allow_pickle=True)

            if data == -1:
                print("Continuing because there aren't a object in the first frame ")
                continue

            input, target = data       

            output = model(input)
            loss_ = loss(output, target)

            loss_mean += loss_.item()

        loss_mean = loss_mean / len(data_loader_test)
        print("Mean loss: ", str(loss_mean))
        viz.plot_lines('test_loss', loss_mean)
        return loss_mean
