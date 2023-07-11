"""
*   dataset is a string passed by command line
"""
def selectDataset(dataset):
    if dataset == 'camnuvem':
        return [49, 437, "CamNuvem_dataset_normalizado"]	# Camnuvem, 49 anomaly samples in test, 437 anomaly samples in training. Third param is the folder name in dataset/
    elif dataset == 'ucf-crime':
        return [140, 810, "ucf_crime_dataset"]	# UCF crime, 140 anomaly samples in test, 810 anomaly samples in trainign
    else:
        print("Dataset not implemented. If you are implementing your own dataset, separate your samples in .list files as in anomalyDetection/files folder, and return here a list as that: [number of anomaly samples in test, number of anomaly samples in training]")
        exit()