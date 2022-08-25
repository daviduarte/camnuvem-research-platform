import visdom
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
import PIL

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


# Run in the temporal graph to localize the 'object predicted' through the all object detected in the self.T frames
def targetFind(object_predicted, adjacency_matrix, frame, SIMILARITY_THRESHOLD, T):

    object_path = np.empty(T)
    object_path.fill(-1)

    next_object = object_predicted

    object_path[0] = int(next_object)

    for i in range(len(adjacency_matrix)):
        
        # if between any frame there was no deteced object, then the object exits the scene
        # So we have to map the NN to EXIT_TOKEN
        if len(adjacency_matrix[i]) == 0:
            return -1, object_path
        obj = torch.argmax(adjacency_matrix[i], dim=1)    

        # If the similarity of two objects in two consecutive frames are less than SIMILARITY_THRESHOLD,
        # we consider they aren't the same. 
        most_probably_next_object_index = obj[next_object]

        if adjacency_matrix[i][next_object][most_probably_next_object_index] < SIMILARITY_THRESHOLD:
            return -1, object_path

        next_object = most_probably_next_object_index
        object_path[i+1] = int(next_object)

    
    # Given a object in the first frame, self.T frames after this object was detected in the 'next_object' index of the object list
    return next_object, object_path


def calculeTarget(adj_mat, bbox_fea_list, box_list, reference_frame, obj_predicted, temporal_graph, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N):

    target_index, object_path = targetFind(obj_predicted, adj_mat, reference_frame, SIMILARITY_THRESHOLD, T)

    # If object exits the scene, lets map the NN to EXIT_TOKEN
    if target_index == -1:
        target = torch.zeros(EXIT_TOKEN)
    else:
        object_in_the_last_frame = bbox_fea_list[-1][1][target_index]        # Last element, objects from last image, object num. 'obj_predicted'
        object_in_the_last_frame_box = box_list[-1][1][target_index]        # a 4 element list

        # The output is the x1, y1, x2, y2 object coordinates concat with object feature
        target = np.append(object_in_the_last_frame_box, object_in_the_last_frame)
        target = torch.from_numpy(target).to(DEVICE)

    # The input will be N-1 objects
    input_boxes = []
    input_features = []
        
    num_obj = len(bbox_fea_list[0][0])
    for i in range(N):
        input_boxes.append(box_list[reference_frame][0][i])
        input_features.append(bbox_fea_list[reference_frame][0][i])

    input_boxes = np.stack(input_boxes, axis=0)
    input_features = np.stack(input_features, axis=0)

    input = np.append(input_boxes, input_features)
    """
        # The input is the (x1, y1, x2, y2) * N-1 object coordinates concat with (object feature) * N-1
        input = bbox_fea_list[reference_frame][0][i]    # A feature
        input = np.append(box_list[reference_frame][0][obj_predicted], input)        
    """
    input = torch.from_numpy(input).to(DEVICE)

    return [input, target], object_path    


def fileLines2List(file):
    # Using readlines()
    file1 = open(file, 'r')
    Lines = file1.readlines()
      
    count = 0
    list_ = []
    # Strips the newline character
    for line in Lines:
        count += 1
        list_.append(line.strip()) 

    return list_


# pred2 = (boxes2, scores2, labels2, bbox_fea_vec2) idem for pred1
def print_image(input, bbox_list, object_path, index):

    str_labels = np.asarray(fileLines2List("coco_labels.txt"))

    #print(len(bbox_list[0]))    

    #print(len(input)-1)
    #exit()


    cont = 0
    #for i in range(len(bbox_list)):
    #    for j in range(len(bbox_list[i])):

            


    for i in range(len(input)):

        image = input[i]
        image = image.cpu()

        object_interest = int(object_path[i])

        # If the object exits of scene, we will not print then
        if object_interest == -1:
            return

        # if we have 4 images, we have 3 bbox_list (each in th middle of each image pair),
        # containing the objects of the anterior (index 0) and posterior (index 1).
        # Yes, there are oject information duplicated
        if i == len(input)-1:
            box = bbox_list[i-1][1][object_interest]
        else:
            box = bbox_list[i][0][object_interest]

        boxes = np.asarray([box])

        #print(boxes.shape)

        image_tensor = torch.from_numpy(np.array(image))
        image_tensor = torch.moveaxis(image_tensor, 2, 0)
        boxes = torch.from_numpy(boxes)

        labels = ['1']

        img_com_bb = draw_bounding_boxes(image_tensor, boxes, labels)
        img_com_bb = torch.moveaxis(img_com_bb, 0, 2)
        img_com_bb = img_com_bb.numpy()
        PIL.Image.fromarray(img_com_bb).convert("RGB").save("imagens/amostra-"+str(index)+"-"+str(cont)+".png")

        cont += 1
    

def img2bbox(input, temporal_graph, N_DOWNSTRAM):

    oi = input[0]
    print("oi  shape")
    print(oi.shape)

    batch_size = oi.shape[0]
    window_frame = oi.shape[1]


    res = []
    box_list = []
    for i in range(batch_size):
        batch = oi[i]
        print("Dinsion inside looping 1")
        print(batch.shape)
        for j in range(window_frame):
            img = oi[i,j]
            print("Dinsion inside looping 2")
            print(img.shape)            
            prediction = temporal_graph.inference(img)
            boxes, scores, labels, bbox_fea_vec  = temporal_graph.filterLowScores(prediction)

            print(bbox_fea_vec.shape)
            if bbox_fea_vec.shape[0] < N_DOWNSTRAM:
                return -1, -1
            print(bbox_fea_vec[0:N_DOWNSTRAM].shape)
            res.append(bbox_fea_vec[0:N_DOWNSTRAM])
            box_list.append(boxes[0:N_DOWNSTRAM])


    
    print("kkk")
    print(res[0].shape)
    res = np.stack(res, axis=0)
    box_list = np.stack(box_list, axis=0)
    box_list = torch.from_numpy(box_list)
    box_list = box_list.view(oi.shape[0], oi.shape[1], N_DOWNSTRAM, 4)
    
    print(res.shape)
    res = torch.from_numpy(res)
    res = res.view(oi.shape[0], oi.shape[1], N_DOWNSTRAM, bbox_fea_vec.shape[1])
    
    print(res.shape)
    print(box_list.shape)

    return res, box_list