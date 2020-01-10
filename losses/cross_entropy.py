import torch.nn.functional as F

class cross_entropy(object):
	def __call__(self,output,target):
		return F.cross_entropy(output,target)
