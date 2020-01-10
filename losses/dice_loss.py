import torch

class dice_loss(object):
    def __call__(self,output,target):
        output,_=torch.max(output,dim=1)
        batch_size=output.shape[0]
        numerator=torch.sum(output*target,dim=(1,2))

        denominator=torch.sum((output**2),dim=(1,2))+torch.sum((target**2),dim=(1,2))
        loss=1-(torch.sum(torch.div(numerator,denominator))/batch_size)
        return loss
