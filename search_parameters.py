import shlex
import subprocess

def writeAlpha(alpha):
	file = open("actual_alpha.txt", 'w')
	file.write(str(alpha))
	file.close()

def writeMargin(margin):
	file = open("actual_margin.txt", 'w')
	file.write(str(margin))
	file.close()

def writeSmooth(smooth):
	file = open("actual_smooth.txt", 'w')
	file.write(str(smooth))
	file.close()	

def writeSparsity(sparsity):
	file = open("actual_sparsity.txt", 'w')
	file.write(str(sparsity))
	file.close()		

with open('parameter_alpha.txt') as file:
    alpha_list = file.readlines()

with open('parameter_margin.txt') as file:
    margin_list = file.readlines()

with open('parameter_smooth.txt') as file:
    smooth_list = file.readlines()

with open('parameter_sparsity.txt') as file:
    sparsity_list = file.readlines()    

for alpha in alpha_list:
	for margin in margin_list:
		for smooth in smooth_list:
			for sparsity in sparsity_list:

				writeAlpha(alpha)
				writeMargin(margin)
				writeSmooth(smooth)
				writeSparsity(sparsity)
				

				command = "python3 run.py --no-feature-extraction=True --make-list-file=False --segment-size=16 --feature-root-train-abnormal=/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/i3d/training/anomaly/ --feature-root-train-normal=/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/i3d/training/normal/ --feature-root-test-abnormal=/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/i3d/test/anomaly/ --feature-root-test-normal=/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/i3d/test/normal/ --make-gt=False --test-labels=/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/dataset/videos/labels/test.txt --video-root-train-normal=/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/videos/samples/training/normal --video-root-test-normal=/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/videos/samples/test/normal --video-root-train-abnormal=/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/videos/samples/training/anomaly --video-root-test-abnormal=/media/denis/526E10CC6E10AAAD/CamNuvem/dataset/videos/samples/test/anomaly --gt=/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/RTFM/list/gt-camnuvem.npy --re-run-test=False --checkpoint=/media/denis/526E10CC6E10AAAD/CamNuvem/pesquisa/anomalyDetection/RTFM/ckpt/rtfm155-i3d_best_camnuvem_dataset_producao.pkl --root=/media/denis/526E10CC6E10AAAD/CamNuvem --make-hd5-file=False --crop-10=False"
				command = shlex.split(command)

				ssh = subprocess.run(command)
 
## Send ssh commands to stdin
#ssh.stdin.write("uname -a\n")
#ssh.stdin.write("uptime\n")
#ssh.stdin.close()

# Fetch output
#for line in ssh.stdout:
#    print(line.strip())