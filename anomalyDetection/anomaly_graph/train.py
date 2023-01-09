import os
import torch
import numpy as np
import definitions
import sys
print(os.path.join(definitions.ROOT_DIR, '../graph_detector'))
sys.path.append(os.path.join(definitions.ROOT_DIR, '../graph_detector'))
import temporalGraph
import util.utils

import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import colorsys

import trajectory_analysis

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def train(normal_dataset, DEVICE, buffer_size, reference_frame, OBJECTS_ALLOWED, N, T, STRIDE, SIMILARITY_THRESHOLD, KEY_FRAME_SIM, GLOBAL_GRAPH):
    print("Iniciando treinamento")
    normal_iter = iter(normal_dataset)

    # Verify if there is a 'cache' folder with files in the definitions.DATASET_DIR folder
    
    has_cache = False
    if os.path.exists(os.path.join(definitions.FRAMES_DIR, "cache/training/normal")):
        has_cache = True
        normal_dataset.has_cache = True
    print("has cache? " + str(has_cache))

    for am in range(len(normal_iter)):
        # Expected input shape: [BATCH, 200]

        # If folder doesn't exist, lets use the dataset and process the frames
        try:
            data = next(normal_iter)   
        except:
            print("Começando de novo")
            normal_iter = iter(normal_dataset)

        input = torch.squeeze(data[0]).to(DEVICE)
        labels = data[1].to(DEVICE)
        folder_index = data[2]
        sample_index = data[3]

        fea_path = os.path.join(definitions.FRAMES_DIR, "cache/training/normal/", str(folder_index.numpy()[0]), str(sample_index.numpy()[0])+"_features.npy")
        graph_path = os.path.join(definitions.FRAMES_DIR, "cache/training/normal/", str(folder_index.numpy()[0]), str(sample_index.numpy()[0])+"_graph.npy")

        if os.path.exists(fea_path) and os.path.exists(graph_path):
            has_cache = True
            normal_dataset.has_cache = True
        else:
            has_cache = False
        print("has cache? " + str(has_cache))        

        if not has_cache:

            print("Não temos a cache, vamos criar")
            temporal_graph = temporalGraph.TemporalGraph(DEVICE, buffer_size, OBJECTS_ALLOWED, N, STRIDE)
            adj_mat, bbox_fea_list, box_list, score_list = temporal_graph.frames2temporalGraph(input, folder_index, sample_index)
            graph_nor = util.utils.calculeTargetAll(adj_mat, bbox_fea_list, box_list, score_list, reference_frame, SIMILARITY_THRESHOLD, T, N)

            path = os.path.join(definitions.FRAMES_DIR, "cache/training/normal/", str(folder_index.numpy()[0]))
            os.makedirs(path, exist_ok=True)

            np.save(fea_path, np.asarray([adj_mat, bbox_fea_list, box_list, score_list]))
            np.save(graph_path, graph_nor)
        else:
            print("Temos a cache, usar xDDDDD")
            adj_mat, bbox_fea_list, box_list, score_list = np.load(fea_path, allow_pickle=True).tolist()
            graph_nor = np.load(graph_path, allow_pickle=True)


        
        # Else, if folder exists, lets read the already processed result
        EXIT_TOKEN = 1  # We don't use it here, but the function waits it
        
        # Run each trajectory in the graph
        last_len = 0
        for i in range(len(graph_nor)):
            size = len(graph_nor[i])

            if size <= last_len:
                continue

            #new_graph = graph_nor[i:]
            for j in range(len(graph_nor[i][last_len:])):
                obj_predicted = j + last_len
                path = trajectory_analysis.analysis.calculeObjectPath(graph_nor, i, obj_predicted)
                #_, object_path_nor = util.utils.calculeTarget(new_graph, score_list, bbox_fea_list, box_list, reference_frame, obj_predicted, DEVICE, EXIT_TOKEN, SIMILARITY_THRESHOLD, T, N)            

                # The object has to have at minimum two frames
                if len(path) <= 1:
                    continue

                # Calcule frame similarity with predecessor frame
                key_frames, _ = trajectory_analysis.analysis.calculeFrameSimilarity(i, path, score_list, bbox_fea_list, box_list, KEY_FRAME_SIM, DEVICE)
                addInGraph(key_frames, score_list, bbox_fea_list, box_list, KEY_FRAME_SIM, GLOBAL_GRAPH, DEVICE)    

            last_len = size

        print("Foram inseridas %s trajetórias" % last_len)

        # Generating sample data
        #G = nx.florentine_families_graph()
    #plot_graph(GLOBAL_GRAPH[1])

def plot_graph(adjacency_matrix):
  # Get the number of nodes in the graph
  num_nodes = len(adjacency_matrix)

  weights_list = []
  for i in range(num_nodes):
    for j in range(num_nodes):
        if j < i:
            weights_list.append(0)
        else:
            weights_list.append(adjacency_matrix[i][j])

  max_ = max(weights_list)
  normalized_values = [float(v) / max_ for v in weights_list]
  hsl_colors = [colorsys.rgb_to_hls(v, v, v) for v in normalized_values]
  hex_colors = [colorsys.hls_to_rgb(h, l, s) for h, l, s in hsl_colors]
  hex_colors = np.array(hex_colors)
  hex_colors = hex_colors.reshape(num_nodes,num_nodes,3)



  # Create a figure and a subplot
  fig, ax = plt.subplots()

  # Generate random x and y coordinates for each node
  x = np.random.uniform(0, 100, size=num_nodes)
  y = np.random.uniform(0, 100, size=num_nodes)

  # Check if there are any edges in the graph
  has_edges = False
  for i in range(num_nodes):
    for j in range(num_nodes):
      if adjacency_matrix[i][j] != 0:
        has_edges = True
        break

  # If there are no edges, simply plot the nodes using the random coordinates
  if not has_edges:
    for i in range(num_nodes):
      # Plot the node
      ax.scatter(x[i], y[i])

  # If there are edges, plot the nodes and the edges using the random coordinates
  else:
    for i in range(num_nodes):
      for j in range(num_nodes):
        # If there is an edge between node i and node j
        if adjacency_matrix[i][j] != 0:
          # Plot a line between the two nodes
          thickness = adjacency_matrix[i][j]
          color = hex_colors[i,j]

          color = matplotlib.colors.to_hex(hex_colors[i,j], keep_alpha=False)

          ax.plot([x[i], x[j]], [y[i], y[j]], color="#0f0f0f80", linewidth=3)

    # Plot the nodes on top of the edges
    for i in range(num_nodes):
      # Plot the node
      ax.scatter(x[i], y[i])

  # Show the plot
  plt.show()


def addInGraph(key_frames, score_list, bbox_fea_list, box_list, KEY_FRAME_SIM, GLOBAL_GRAPH, DEVICE):

    # compare all key frames with all nodes in global graph
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08).to(DEVICE)    

    # We nee to verify if one path have same keyframe in different momments
    vec1_ = torch.repeat_interleave(key_frames, key_frames.shape[0], dim=0).to(DEVICE)
    vec2_ = key_frames.repeat(key_frames.shape[0], 1).to(DEVICE)
    appea_dist_path = cos(vec1_, vec2_).view(key_frames.shape[0], key_frames.shape[0])

    # Initialy, each key frame is mapped to your respectivelly feture vector in 'key_frame'
    map_key_frame = [i for i in range(len(key_frames))]
    for i in range(len(appea_dist_path)):
        for j in range(i+1, len(appea_dist_path)):
            if appea_dist_path[i,j] >= KEY_FRAME_SIM:
                # se cair aqui sei que o [i, j] possui o esmo key frame do [i,i]
                map_key_frame[j] = map_key_frame[i]

    # this is the key frame we have to add
    unique_key_frames_index = np.unique(map_key_frame)
    
    # Spare the nodes used in object path
    key_frames_sublist = key_frames[unique_key_frames_index]

    # Ok, now let's verify if the nodes in 'key_frames_sublist' already exists in 'vertex'
    #vertex.append(key_frames[0].tolist())
    #vertex.append(key_frames[1].tolist())

    if len(key_frames_sublist) == 0:
        print("O path não possui key frames. Ele está vazio? Tamanho do path: " + str(len(key_frames)))
        exit()

    vertex = GLOBAL_GRAPH[0]
    if len(vertex) == 0:
        node_index = add_node(GLOBAL_GRAPH, key_frames_sublist[0])

    vertex = GLOBAL_GRAPH[0]
    vertex = torch.FloatTensor(vertex)

    #key_frames_sublist = torch.from_numpy(key_frames_sublist)

    vec1_ = torch.repeat_interleave(vertex, key_frames_sublist.shape[0], dim=0).to(DEVICE)
    vec2_ = key_frames_sublist.repeat(vertex.shape[0], 1).to(DEVICE)

    appea_dist = cos(vec1_, vec2_).view(key_frames_sublist.shape[0], vertex.shape[0])
            
    nodes_to_add = appea_dist.shape[0]
    nodes_in_vextex_num = appea_dist.shape[1]
    #KEY_FRAME_SIM = 0.9999999
    
    # I need a map from unique_key_frames_index to vertex indexes
    map_to_vertex = [-1 for i in range(unique_key_frames_index.shape[0])]
    for i in range(nodes_to_add):
        exists = False
        node_index = False
        for j in range(nodes_in_vextex_num):
            #print("Sim %s %s: %s" % (i,j,appea_dist[i,j]))
            if appea_dist[i,j] > KEY_FRAME_SIM:      # IF the node already exists in VEXTEX
                exists = True
                node_index = j
                break

        if not exists:
            node_index = add_node(GLOBAL_GRAPH, key_frames_sublist[i])
        
        map_to_vertex[i] = node_index
    

    # Agora temos que atribuir os pesos às arestas 
    for i in range(1, len(key_frames)):        # Para cada transição de key_frame no path

        before = i-1
        after = i
        index_before = np.where(unique_key_frames_index == map_key_frame[before])
        index_after = np.where(unique_key_frames_index == map_key_frame[after])
        if len(index_before[0])==0 or len(index_after[0]) == 0:
            print("Ops, deu erro")
            exit()
        index_before = index_before[0][0]
        index_after = index_after[0][0]


        index_in_graph_before = map_to_vertex[index_before]
        index_in_graph_after = map_to_vertex[index_after]

        # Put this path in edges in graph
        adj_matrix = GLOBAL_GRAPH[1]
        #adj_matrix[index_before][index_after] += 1
        adj_matrix[index_in_graph_before][index_in_graph_after] += 1



def add_node(GLOBAL_GRAPH, node):

    adj_matrix = GLOBAL_GRAPH[1]
    vertex = GLOBAL_GRAPH[0]

    vertex.append(node.tolist())
    if len(adj_matrix) == 0:
        adj_matrix.append([0])
        return 0
    
    # Add a new row and column to the adjacency matrix
    adj_matrix.append([0] * (len(adj_matrix)+1))
    for row in adj_matrix[:-1]:
        row.append(0)

    # Return the updated adjacency matrix
    return len(adj_matrix)-1
    
