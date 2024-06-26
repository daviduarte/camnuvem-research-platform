"""
Davi Duarte, UNESP, Prisma
This code extract feature vectors from videos and train a anomaly detector
"""

import configparser
import numpy as np
import option
import importlib  
import os
import sys
ROOT_DIR = os.path.abspath(os.curdir)
#sys.path.append("/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa")
#sys.path.append("/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/extractI3d/pytorch-resnet3d/models")
sys.path.append(os.path.join(ROOT_DIR, "extractI3d/pytorch-resnet3d"))
#sys.path.append("/home/denis/Documentos/CamNuvem/pesquisa/anomalyDetection/RTFM")
sys.path.append(os.path.join(ROOT_DIR, "anomalyDetection"))
import extractI3d
from anomalyDetection.WSAL import create_hd5
from datasetConfig import selectDataset

from RTFM import main as RTFM_Train
from RTFM.list import make_list_camnuvem
from WSAL import Train as WSAL_Train
from RADS import main as RADS_Train

#meu_main = importlib.import_module("extractI3d.pytorch-resnet3d.meu_main")
#from extractI3d.pytorch-resnet3d import meu_main

def sanityCheck(args):

	# These two cannot be true in same time. 
	if args.no_anomaly_detection == True and args.no_feature_extraction == True:
		print("You have to do at least --no_anomaly_detection or --no_feature_extraction")
		exit()

if __name__ == '__main__':

	args = option.parser.parse_args()
	sanityCheck(args)
	test_labels = args.test_labels
	anomalyDetectionMethod = args.anomaly_detection_method
	dataset = args.dataset
	datasetInfos = selectDataset(dataset)
	print("You are using "+anomalyDetectionMethod+" as video surveillance anomaly detection method. ")
	print("You are using "+dataset+" dataset. It uses " + str(datasetInfos[0]) + " anomaly samples for testing and " + str(datasetInfos[1]) + " anomaly samples for training.")

	if args.root == "False":
		print("Root argument mandatory")
		exit()
	root = args.root
	num_frames_in_each_feature_vector= args.segment_size

	example_video_path = []
	#example_video_path.append("/home/denis/Documentos/CamNuvem/violent_thefts_dataset/videos/samples/test/anomaly/")
	#example_video_path.append("0.mp4")


	if args.crop_10 == "True":
		crop_10 = True			
	else:
		crop_10 = False

	"""
		Make the feature extraction
	"""
	if args.no_feature_extraction == "False":

		feature_extractor = args.feature_extractor

		video_root_test_abnormal = args.video_root_test_abnormal
		video_root_train_abnormal = args.video_root_train_abnormal
		video_root_test_normal = args.video_root_test_normal
		video_root_train_normal = args.video_root_train_normal


		feature_root_test_abnormal = args.feature_root_test_abnormal
		feature_root_train_abnormal = args.feature_root_train_abnormal
		feature_root_test_normal = args.feature_root_test_normal
		feature_root_train_normal = args.feature_root_train_normal

		paths = [[video_root_test_abnormal, feature_root_test_abnormal], 
				[video_root_train_abnormal, feature_root_train_abnormal], 
				[video_root_test_normal, feature_root_test_normal], 
				[video_root_train_normal, feature_root_train_normal]]

		for path in paths:

			print(path)
			path_to_load_videos = path[0]
			path_to_save_npy = path[1]

			if path_to_load_videos == "False" or path_to_save_npy == "False":
				print("SKIPANDO*****")
				continue

			print("\n*** Extracting features from videos in " + path_to_load_videos + " and putting the resulting extracted feature in " + path_to_save_npy + "\n\n")
			video_path = ["", ""]
			for filename in os.listdir(path_to_load_videos):



				video = os.path.join(path_to_load_videos, filename)
				video_path[0] = path_to_load_videos
				video_path[1] = filename


				npy_name_file = video_path[1][:-4]+".npy"
				npy_name_file = path_to_save_npy + npy_name_file

				if args.replace == "False" and os.path.exists(npy_name_file):
					print("O arquivo " + path_to_save_npy + " já existe. Continuando...")
					continue		

				extractI3d.video2npy(video_path, path_to_save_npy, crop_10, feature_extractor, args.gpu_id) 


	"""
		Make the anomaly detection training
	"""
	if args.no_anomaly_detection == "False":

		#test_list_file_final_name = args.test_list_file_final_name
		#training_list_file_final_name = args.training_list_file_final_name	

		if args.feature_extractor == "False":
			print("Escolha um extrator de características com a flag --feature-extractor")
			exit()

		if crop_10:
			test_list_file_final_name = os.path.join(root, "pesquisa/anomalyDetection/files/"+dataset+"-"+args.feature_extractor+"-test-10crop.list")
			training_list_file_final_name = os.path.join(root, "pesquisa/anomalyDetection/files/"+dataset+"-"+args.feature_extractor+"-train-10crop.list")
			teste_only_abnormal_file_final_name = os.path.join(root, "pesquisa/anomalyDetection/files/"+dataset+"-"+args.feature_extractor+"-test-abnormal-only-10crop.list")
		else:
			test_list_file_final_name = os.path.join(root, "pesquisa/anomalyDetection/files/"+dataset+"-"+args.feature_extractor+"-test.list")
			training_list_file_final_name = os.path.join(root, "pesquisa/anomalyDetection/files/"+dataset+"-"+args.feature_extractor+"-train.list")					
			teste_only_abnormal_file_final_name = os.path.join(root, "pesquisa/anomalyDetection/files/"+dataset+"-"+args.feature_extractor+"-test-abnormal-only.list")			


		# Verify is we have to create the list file. This is mandatory either by RTFM than WSAL
		if args.make_list_file == "True":

			# Mandatory arguments for RTFM and WSAL
			#if args.test_list_file_final_name == "False" or args.training_list_file_final_name == "False":
			#	print("args.test_list_file_final_name and args.training_list_file_final_name are mandatory")
			#	exit()

			#final_name = args.list_file_final_name
		
			i3d_root_train_abnormal = args.feature_root_train_abnormal
			i3d_root_train_normal = args.feature_root_train_normal
			i3d_root_test_abnormal = args.feature_root_test_abnormal
			i3d_root_test_normal = args.feature_root_test_normal
			if (i3d_root_train_abnormal=="False" or i3d_root_train_normal=="False" or i3d_root_test_abnormal=="False" or i3d_root_test_normal=="False"):
				print("You need enter the path of normal and abnormal train and test files. See --help")
				exit()


			make_list_camnuvem.make_list_file(test_list_file_final_name, training_list_file_final_name, teste_only_abnormal_file_final_name, i3d_root_train_abnormal, i3d_root_train_normal, i3d_root_test_abnormal, i3d_root_test_normal, test_labels)

		if anomalyDetectionMethod == "RTFM":
			

			# Mandatory arguments for RTFM
			#if args.test_list_file_final_name == "False" or args.training_list_file_final_name == "False":
			#	print("args.test_list_file_final_name and args.training_list_file_final_name are mandatory")
			#	exit()

			args.test_rgb_list = test_list_file_final_name
			args.rgb_list = training_list_file_final_name
	


			# TODO: pass all arguments in args to subrotine arguments
			RTFM_Train.train(args, test_list_file_final_name, datasetInfos)

		elif anomalyDetectionMethod == "WSAL":

			if args.feature_extractor == "False":
				print("Escolha um extrator de características com a flag --feature-extractor")
				exit()

			if crop_10:
				hd5_train = os.path.join(root, "pesquisa/anomalyDetection/files/data_train_"+args.feature_extractor+"_10crop.h5")
				hd5_test = os.path.join(root, "pesquisa/anomalyDetection/files/data_test_"+args.feature_extractor+"_10crop.h5")
			else:
				hd5_train = os.path.join(root, "pesquisa/anomalyDetection/files/data_train_"+args.feature_extractor+".h5")
				hd5_test = os.path.join(root, "pesquisa/anomalyDetection/files/data_test_"+args.feature_extractor+".h5")


			if args.make_hd5_file=="True":
				# WSAL need dataset be in h5 format. 				
				create_hd5.run(root, test_list_file_final_name, training_list_file_final_name, crop_10, hd5_train, hd5_test, datasetInfos)

			print("Trainando o WSAL")
				
			WSAL_Train.train_wsal(training_list_file_final_name, test_list_file_final_name, hd5_train, hd5_test, num_frames_in_each_feature_vector, root, crop_10, args.gpu_id, args.checkpoint, datasetInfos)

	
		elif anomalyDetectionMethod == "RADS":
			RADS_Train.main(training_list_file_final_name, test_list_file_final_name, num_frames_in_each_feature_vector, root, crop_10, args.gpu_id, args.checkpoint, datasetInfos, args.feature_extractor)