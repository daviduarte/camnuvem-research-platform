import torch
import torch.nn as nn
import torch.nn.init as torch_init


# TODO:
# Weight init
class ModelPretext(nn.Module):
	def __init__(self, num_feat_in, num_feat_out):
		super(ModelPretext, self).__init__()	
		print("num_feat_in: ")
		print(num_feat_in)
		print("num_feat_out: ")
		print(num_feat_out)		
		self.fc1 = nn.Linear(num_feat_in, 512)
		self.fc2 = nn.Linear(512, 128)
		self.fc3 = nn.Linear(128, num_feat_out)   

		self.relu = nn.ReLU()  

	def forward(self, inputs):
		fc1 = self.relu(self.fc1(inputs))
		fc2 = self.relu(self.fc2(fc1))

		# In the downstream, the target is the object resnet feature vector. This is in the range [0-255] because is uint8
		# So, relu may do the work in the last layer		
		fc3 = self.relu(self.fc3(fc2))

		return fc3
