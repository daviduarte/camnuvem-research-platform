from RTFM.model import Model
import numpy as np
import torch
import torchvision

THRESHOLD = 0.5		# TODO: Verificar o melhor THRESHOLD no test do artigo
FEATURE_SIZE = 2048
BATCH_SIZE = 32

CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE == True:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

def inferencia(feature_extractor):

	#input = np.load(feature_extractor)
	#print(input.shape)
	#input = np.moveaxis(input, 0, 1)	
	#print(input.shape)

	input = feature_extractor
	input = np.expand_dims(input, axis=0)
	input = np.expand_dims(input, axis=0)
	input = torch.from_numpy(input)


	#input = #[1, 10, 89, 2048]

	model = Model(FEATURE_SIZE, BATCH_SIZE)
	checkpoint = torch.load("/media/davi/dados/Projetos/CamNuvem/pesquisa/RTFM/ckpt/ucf-i3d-ckpt.pkl", map_location=torch.device('cpu'))
	model.load_state_dict(checkpoint)

	score_abnormal, score_normal, feat_select_abn, feat_select_normal, feat_abn_bottom, feat_select_normal_bottom, logits_, \
	scores_nor_bottom, scores_nor_abn_bag, feat_magnitudes = model(inputs=input) 


	# retorna 0 ou 1, que indica se um segmento é anômalo ou não
	classes = []
	for i in range(logits_.shape[1]):
		valor = logits_[0,i,0].detach().numpy()	# Pega o valor bruto do tensor e coloca em uma lista
		if valor > 0.7399341:
			classes.append(1)
		else:
			classes.append(0)

	return classes
