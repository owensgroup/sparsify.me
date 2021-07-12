#%%
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.nn.functional import unfold
import math
import csv
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%% get model


N, C, H, W = 32, 3, 224, 224
batch = torch.randn(N, C, H, W).to(device)

def calculate_output_size(inp_shape, kernel_size, stride=1, padding=0, dilation=1):
    return math.floor(((inp_shape + 2 * padding - dilation * (kernel_size - 1) - 1)/stride) + 1)

def print_resnet_conv_shapes(model):
    with torch.no_grad():
        input_img = torch.randn(32, 3, 224, 224).to(device)
        matmul_shapes = []
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d) and ('downsample' not in name):
                print(name)
                unfolded = unfold(input_img,
                    kernel_size=layer.kernel_size,
                    dilation=layer.dilation,
                    padding=layer.padding,
                    stride=layer.stride)
                filters = layer.weight.view(layer.weight.size(0), -1)
                shapes = (filters.shape, unfolded.shape)
                print(shapes)
                matmul_shapes.append(shapes)
                out = (filters @ unfolded)
                out_h = calculate_output_size(input_img.shape[2], layer.kernel_size[0], layer.stride[0], layer.padding[0], layer.dilation[0])
                out_w = calculate_output_size(input_img.shape[3], layer.kernel_size[1], layer.stride[1], layer.padding[1], layer.dilation[1])
                input_img = out.view(out.size(0), out.size(1), out_h, out_w)
        return matmul_shapes

def print_mobilenet_shapes(model):
  input_img = torch.randn(32, 3, 244, 244).to(device)
  matmul_shapes = []
  with torch.no_grad():
    for name, layer in model.named_modules():
      if isinstance(layer, torch.nn.modules.conv.Conv2d):
        unfolded = unfold(input_img,
                    kernel_size=layer.kernel_size,
                    dilation=layer.dilation,
                    padding=layer.padding,
                    stride=layer.stride)
        filters = layer.weight.view(layer.weight.size(0), -1)
        shapes = (filters.shape, unfolded.shape)
        print(name)
        print(shapes)
        matmul_shapes.append(shapes)
        out = (filters @ unfolded)
        out_h = calculate_output_size(input_img.shape[2], layer.kernel_size[0], layer.stride[0], layer.padding[0], layer.dilation[0])
        out_w = calculate_output_size(input_img.shape[3], layer.kernel_size[1], layer.stride[1], layer.padding[1], layer.dilation[1])
        input_img = out.view(out.size(0), out.size(1), out_h, out_w)
      else:
        input_img = layer(input_img)
  return matmul_shapes

def write_shapes(filename, shape_list):
  with open(filename + ".csv", mode='w') as shape_file:
    shape_writer = csv.writer(shape_file, delimiter=',')
    shape_writer.writerow(['m','n','k','b'])
    for s in shape_list:
      st = [str(s[1][2]), str(s[0][0]), str(s[0][1]), str(s[1][0])]
      shape_writer.writerow(st)

def profile_conv_runtimes(model, filename):
  model = model.cuda()
  inputs = torch.randn(32, 3, 224, 224).cuda()
  with profile(activities=[ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    with record_function("model_inference"):
      model(inputs)
  print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))
  print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=10))
  print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_memory_usage", row_limit=10))
  prof.export_chrome_trace(filename + '.json')
# %%
model_zoo = {
  'resnet18': models.resnet18(),
  'resnet34': models.resnet34(),
  'resnet50': models.resnet50(),
  'resnet101': models.resnet101(),
  'resnet152':models.resnet152(),
  'mobilenetv2': models.mobilenet.mobilenet_v2(),
  'mobilenetv3_small': models.mobilenet.mobilenet_v3_small(),
  'mobilenetv3_large': models.mobilenet.mobilenet_v3_large(),
  'densenet161': models.densenet161(),
  'densenet201': models.densenet201()
}

profile_conv_runtimes(models.mobilenet.mobilenet_v3_small(), 'resnet18')

# %%
model = models.mobilenet.mobilenet_v2()
model = model.to(device)
print_mobilenet_shapes(model)
  # shapes = print_resnet_conv_shapes(model)
  # write_shapes(name, shapes)
# %%
