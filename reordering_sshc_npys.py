import os
import numpy as np
dirs = os.listdir("/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames_05s/test/anomaly")
dirs.sort()
npys = [i+1 for i in range(len(dirs))]
print("dirs list:")
print(dirs)

labels = open("/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/labels/test.txt", 'r')
labels = [label.strip().split(" ")[0][:-4] for label in labels.readlines()]
print("labels list: ")
print(labels)

new_list = ["/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado/sshc/test/anomaly/"+str(npys[dirs.index(labels[i])])+".npy" for i in range(len(labels))]

print("new list")
print(new_list)
exit()

file = open("/media/denis/dados/CamNuvem/pesquisa/anomalyDetection/files/camnuvem-yolov5-test-reordered.list", 'w')
for line in new_list:
	file.write(line+"\n")
file.close()


exit()

list_file = "/media/denis/dados/CamNuvem/dataset/CamNuvem_dataset_normalizado_frames_05s/test_anomaly.txt"
list = open(list_file, 'r')
list_lines = [line.strip() for line in list.readlines()]
vec_list = [line.split("/")[2].split("|")[0] for line in list_lines]
new_list_index = [vec_list.index(dirs[npys[i]-1]) for i in range(len(npys))]
noiz_que = np.asarray(list_lines)[new_list_index]
file = open("camnuvem-sshc-test-reordered.list", 'w')
for line in noiz_que:
	file.write(line+"\n")
file.close()
