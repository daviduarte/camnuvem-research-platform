import numpy as np
import h5py
import os


#d1 = np.random.random(size = (5, 10,32,2048))


def run(root, list_test, list_training, ten_crop, arquivo_train, arquivo_test, offset):

	
	"""
	dest_train_normal = "/home/denis/Documentos/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/samples/training/normal/"
	dest_train_anomaly = "/home/denis/Documentos/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/samples/training/anomaly/"
	dest_test_normal = "/home/denis/Documentos/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/samples/test/normal/"
	dest_test_anomaly = "/home/denis/Documentos/CamNuvem/dataset/CamNuvem_dataset_normalizado/videos/samples/test/anomaly/"
	"""

	iter = [list_test, list_training]
	#offset = [49, 437]	# Camnuvem
	#offset = [140, 810]	# UCF crime

	#if ten_crop == True:
	#	arquivo_train = "pesquisa/anomalyDetection/files/data_train_10crop.h5"
	#	arquivo_test = "pesquisa/anomalyDetection/files/data_test_10crop.h5"
	#else:
	#	arquivo_train = "pesquisa/anomalyDetection/files/data_train.h5"
	#	arquivo_test = "pesquisa/anomalyDetection/files/data_test.h5"

	for i in range(2):

		if i == 0:
			print("Criando o list de teste")
			hf = h5py.File(os.path.join(root, arquivo_test), 'w')
		else:
			print("Criando o list de treinamento")
			hf = h5py.File(os.path.join(root, arquivo_train), 'w')

		file = iter[i]
		offset_ = offset[i]

		file = open(file, 'r')
		lines = file.readlines()

		count = 0
		for line in lines:
			path = line.strip()
			name = path.strip().split('/')[-1][:-4]

			features = np.load(path)

			if count < offset_:
				name = "anomaly_"+name
			else:
				name = "normal_"+name

			print(name)
			hf.create_dataset(name, data=features)
			count += 1


		hf.close()



if __name__ == "__main__":

	list_training = "/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/WSAL/arquivos/camnuvem-i3d-train-10crop.list"
	list_test = "/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/WSAL/arquivos/camnuvem-i3d-test-10crop.list"	
	run(list_test, list_training)
