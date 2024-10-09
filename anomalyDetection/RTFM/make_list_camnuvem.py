import numpy as np
import os
import glob

def make_list_file(final_name_test, final_name_train, teste_only_abnormal_file_final_name, i3d_root_train_abnormal, i3d_root_train_normal, i3d_root_test_abnormal, i3d_root_test_normal, test_labels):

	print(" ** Making the list file **")

	with open(final_name_test, 'w+') as f:

		videos_name = [line.strip().split()[0][:-4] for line in open(test_labels)]
		new_file = ["" for i in videos_name]

		files = sorted(glob.glob(os.path.join(i3d_root_test_abnormal, "*.npy")))

		for file in files:
			last_part = file.rsplit('/', 1)[-1]  # Split by the last '/', take the last part
			video_name = last_part[:-4] # Remove the last 4 characters
			index =videos_name.index(video_name)	# Search the right position
			
			newline = file+'\n'
			new_file[index] = newline	# Put the line in the right position
			

		for line in new_file:	
			f.write(line)

		
		# We don't need normal files in order.
		files = sorted(glob.glob(os.path.join(i3d_root_test_normal, "*.npy")))

		for file in files:
			newline = file+'\n'
			f.write(newline)		


	with open(final_name_train, 'w+') as f:

		files = sorted(glob.glob(os.path.join(i3d_root_train_abnormal, "*.npy")))

		for file in files:
			newline = file+'\n'
			f.write(newline)

		files = sorted(glob.glob(os.path.join(i3d_root_train_normal, "*.npy")))

		for file in files:
			newline = file+'\n'
			f.write(newline)		



if __name__ == '__main__':
	i3d_root_train_abnormal = "/media/denis/526E10CC6E10AAAD/CamNuvem/violent_thefts_dataset/i3d/training/anomaly"
	i3d_root_train_normal = "/media/denis/526E10CC6E10AAAD/CamNuvem/violent_thefts_dataset/i3d/training/normal"
	i3d_root_test_abnormal = "/media/denis/526E10CC6E10AAAD/CamNuvem/violent_thefts_dataset/i3d/test/anomaly"
	i3d_root_test_normal = "/media/denis/526E10CC6E10AAAD/CamNuvem/violent_thefts_dataset/i3d/test/normal"


	make_list_file("camnuvem_list_file_final_", i3d_root_train_abnormal, i3d_root_train_normal, i3d_root_test_abnormal, i3d_root_test_normal)