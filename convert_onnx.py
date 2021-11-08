import os
import argparse

import torch
import yaml
from torch import nn
from backbone import EfficientDetBackbone
import numpy as np

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

      
if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-w', '--weights', type=str, default=None, help='Weights file')
    parser.add_argument('-o', '--out_file', type=str, default=None, help='ONNX file name')
    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    params = Params(f'projects/{args.project}.yml')
    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=args.compound_coef, onnx_export=True,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales)).to(device)


    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    model.backbone_net.model.set_swish(memory_efficient=False)

    dummy_input = torch.randn((1, 3, input_sizes[args.compound_coef], input_sizes[args.compound_coef]), dtype=torch.float32).to(device)

    assert os.path.isfile(args.weights), f'Not found weights file {args.weights}'
    model.load_state_dict(torch.load(args.weights))

    # opset_version can be changed to 10 or other number, based on your need
    if args.out_file is None:
        out_file = os.path.splitext(args.weights)[0] + '.onnx'
    else:
        out_file = args.out_file
    torch.onnx.export(model, dummy_input,
                    out_file,
                    verbose=False,
                    input_names=['data'],
                    opset_version=11)
    print(f'Exported ONNX model as {out_file}')
