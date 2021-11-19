import torch
import torch.quantization
import torch.nn as nn
import os

from torch.nn.parallel.data_parallel import DataParallel
from torchvision.models.resnet import resnet50
from torch.quantization import QuantStub, DeQuantStub

# from stacked_hourglass.model import hg2
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# class lstm_for_demonstration(nn.Module):
#   """Elementary Long Short Term Memory style model which simply wraps nn.LSTM
#      Not to be used for anything other than demonstration.
#   """
#   def __init__(self,in_dim,out_dim,depth):
#      super(lstm_for_demonstration,self).__init__()
#      self.lstm = nn.LSTM(in_dim,out_dim,depth)

#   def forward(self,inputs,hidden):
#      out,hidden = self.lstm(inputs,hidden)
#      return out, hidden

class Mtest(nn.Module):
   def __init__(self):
      super().__init__()
      self.quant = QuantStub()
      self.conv1 = nn.Conv2d(3, 50, (3, 3))
      self.dequant = DeQuantStub()
      
   def forward(self, d):
      x = self.quant(d)
      x = self.conv1(x)
      x = self.dequant(x)

      return x

torch.manual_seed(29592)  # set the seed for reproducibility

#shape parameters
# model_dimension=8
# sequence_length=20
# batch_size=1
# lstm_depth=1

# hidden is actually is a tuple of the initial hidden state and the initial cell state

def load_model(path):
    print('Loading model weights from file: {}'.format(path))
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    model = hg2(pretrained=False)
    if sorted(state_dict.keys())[0].startswith('module.'):
        model = DataParallel(model)
    model.load_state_dict(state_dict)

    return model

def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    print("model: ",label,' \t','Size (KB):', size/1e3)
    os.remove('temp.p')
    return size


model = Mtest()
# random data for input
inputs = torch.randn((1, 3, 100, 100))
model(inputs)
 # here is our floating point instance
# float_lstm = lstm_for_demonstration(model_dimension, model_dimension,lstm_depth)
# DOESN'T WORK WITH Conv2D by themselves
# m = resnet50()
# qm = torch.quantization.quantize_dynamic(
#     m, {nn.Conv2d, nn.Linear}, dtype=torch.qint8
# )
# this is the call that does the work
# quantized_lstm = torch.quantization.quantize_dynamic(
#     float_lstm, {nn.LSTM, nn.Linear}, dtype=torch.qint8
# )

# show the changes that were made
# print('Here is the floating point version of this module:')
# print(m)
# print('')
# print('and now the quantized version:')
# print(qm)

# f=print_size_of_model(m,"fp32")
# q=print_size_of_model(qm,"int8")
# print("{0:.2f} times smaller".format(f/q))

# model = load_model('../checkpoint/hg2/checkpoint.pth.tar')
# qmodel = torch.quantization.quantize_dynamic(model, set([torch.nn.Conv2d]), dtype=torch.qint8)

# compare the sizes
# f=print_size_of_model(model,"fp32")
# q=print_size_of_model(qmodel,"int8")
# print("{0:.2f} times smaller".format(f/q))


# print(qmodel)

