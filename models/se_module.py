import torch
import math
from torch import nn
import torch.nn.functional as F
#from torch.autograd import Function

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # print(channel)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )
        #self.resize = resize()

    def forward(self, x):
        b, c, w, h = x.size()
        # print(x.shape)
        # print(x.type())
        out = self.avg_pool(x).view(b, c)
        #out = F.avg_pool2d(x, c).view(b, c)
        out = self.fc(out).view(b, c, 1, 1)

        # c_new = int(math.floor(c*0.5))
        # out, sort_key = torch.sort(out, 1, True)
        # x_new = torch.zeros(b,c_new,w,h) ##iru
        # out_new = torch.zeros(b,c_new,1,1) #iru
        # x_new=x_new.cuda()
        # out_new=out_new.cuda()
        # for i in xrange(b):
        #     x_new[i] = x[i, sort_key[i,0:c_new,0,0],:,:]
        #     out_new[i] = out[i,0:c_new,:,:] ##iru
        # return x_new*out_new
        return x*out

    #def backward(self, grad_output):
    #    result = self.x_new
    #    print(result.shape)
    #    grad_input = grad_output.numpy() * (result.numpy() > 0)
    #    return torch.FloatTensor(grad_input)
