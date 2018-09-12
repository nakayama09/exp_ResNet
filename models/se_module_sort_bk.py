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

        c_new = int(math.floor(c*0.5))
        out, sort_key = torch.sort(out, 1, True)
        x_new = torch.zeros(b,c_new,w,h) ##iru
        # print('out:'+str(out.shape))
        # print('x_new:'+str(x_new.shape))
        # x_new = torch.zeros(b,c,w,h)
        #x_new=x_new.cuda() ##iru
        # out=out.to('cuda') ##iru
        # print(out.type())
        #print('sort'+str(sort_key[0]))
        out_new = torch.zeros(b,c_new,1,1) #iru
        # print(out_new.type())
        #out_new = out_new.cuda()
        # print(out_new.type())
        #z=[[[[]]]]
        #sort_key.resize_(b,c)
        #print(sort_key.shape)
        #print(sort_key[0,:,0,0])
        #print(x[0])
        for i in xrange(b):
            for j in xrange(c_new):
                x_new[i,j] = x[i, sort_key[i,j,0,0],:,:]
            #x[i] = x[i, sort_key[i,:,0,0],:,:]
            #x_new[i]=x_new[i].clone()*out[i]
            #print(c_new)
            #x[i].resize_(c_new,w,h)
            #out[i].resize_(c_new,1,1)
            #x_new[i]=x[i,0:c_new,:,:] ##iru
            #print(x_new.shape)
            out_new[i] = out[i,0:c_new,:,:] ##iru
            #x_new[i]=x_new[i]*out_new[i]
            #print(out_new.shape)
        #print(x[0])
        # x_new=x_new.cuda() ##iru
        # out=out.cuda() ##iru
        # print(out_new.type())
        #print(type(x_new))
        # self.x_new = x.clone()*out.clone() ##iru
        # print('bbbbbbbbbbbbbbbbbbb')
        x_new=x_new.cuda()
        # print('aaaaaaaaaaaaaaaaaaaa')
        out_new=out_new.cuda()
        # print(type(out_new))
        #print(self.x_new.shape)
        # return self.x_new ##iru?
        # print(x_new.type())
        #return x_new
        return x_new*out_new
        #return x*out

    #def backward(self, grad_output):
    #    result = self.x_new
    #    print(result.shape)
    #    grad_input = grad_output.numpy() * (result.numpy() > 0)
    #    return torch.FloatTensor(grad_input)
