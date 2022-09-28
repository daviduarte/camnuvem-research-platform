import torch
import utils
import temporalGraph
import numpy as np



def test(model, loss, test_loader, reference_frame, obj_predicted, viz, DEVICE, EXIT_TOKEN, N, SIMILARITY_THRESHOLD, T, OBJECTS_ALLOWED):    
    print("Testing")
    temporal_graph = temporalGraph.TemporalGraph(DEVICE, OBJECTS_ALLOWED, N)
    data_loader_test = iter(test_loader)
    with torch.no_grad():
        model.eval()

        loss_mean = 0
        for i in range(len(data_loader_test)):
            input = next(data_loader_test)
            input = np.squeeze(input)

            adj_mat, bbox_fea_list, box_list, score_list = temporal_graph.frames2temporalGraph(input)

            SIMILARITY_THRESHOLD = 0.65#0.73
            graph = utils.calculeTargetAll(adj_mat, bbox_fea_list, box_list, score_list, reference_frame, temporal_graph, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)


            # If in the first frame there is no object detected, so we have nothing to do here
            # The number of detected objects may be less than N. In this case we have nothing to do here
            #if len(bbox_fea_list[reference_frame][obj_predicted]) < N:
            #    print("continuando")
            #    continue       # Continue

            data, object_path = utils.calculeTarget(graph, score_list, bbox_fea_list, box_list, reference_frame, obj_predicted, temporal_graph, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)
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
