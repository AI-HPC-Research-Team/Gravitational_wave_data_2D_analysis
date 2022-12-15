import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os
from models import build_model
from config import get_config
import argparse

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config




transform_test = transforms.Compose([
    transforms.Resize((224, 224),
                                  interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.266, 0.031, 0.342), (0.029, 0.082, 0.046))
])
classes = ("0", "1","2","3","4")




DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_, config = parse_option()
for i in config:
    print(i)
model = build_model(config)
checkpoint = torch.load('output/swin_tiny_patch4_window7_224/default/ckpt_epoch_75.pth', map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()
model.to(DEVICE)
true_list = []
pred_list = []
path = '/userhome/renym/data/gwdata_gen/Image-20db/val/'
testList = os.listdir(path)
wrong = open('wrong.txt', 'a')
for dirs in testList:
    for file in os.listdir(path + dirs):
        img = Image.open(path + dirs + '/' +file)
        img = transform_test(img)
        img.unsqueeze_(0)
        img = Variable(img).to(DEVICE)
        out = model(img)
        # Predict
        _, pred = torch.max(out.data, 1)
        if 'bwd_' in file:
            print
            true_list.append(1)
#             print(type(int(classes[pred.data.item()])))
            pred_list.append(int(classes[pred.data.item()]))
            if int(classes[pred.data.item()]) != 1:
                print("*******************************")
                print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
                wrong.write('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]) + '\n')
        if 'noise_' in file:
            true_list.append(0)
            pred_list.append(int(classes[pred.data.item()]))
            if int(classes[pred.data.item()]) != 0:
                print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
                wrong.write('Image Name:{},predict:{}'.format(file, classes[pred.data.item()])+ '\n')
        if 'sgwb_' in file:
            true_list.append(3)
            pred_list.append(int(classes[pred.data.item()]))            
            if int(classes[pred.data.item()]) != 3:
                print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
                wrong.write('Image Name:{},predict:{}'.format(file, classes[pred.data.item()])+ '\n')
        if 'emri_' in file:
            true_list.append(2)
            pred_list.append(int(classes[pred.data.item()]))            
            if int(classes[pred.data.item()]) != 2:
                print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
                wrong.write('Image Name:{},predict:{}'.format(file, classes[pred.data.item()])+ '\n')
        if 'smbhb_' in file:
            true_list.append(4)
            pred_list.append(int(classes[pred.data.item()]))            
            if int(classes[pred.data.item()]) != 4:
                print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))
                wrong.write('Image Name:{},predict:{}'.format(file, classes[pred.data.item()])+ '\n')
from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
import numpy as np
np_true = np.array(true_list)
np_pred = np.array(pred_list)
wrong.close()
cm = confusion_matrix(np_true, np_pred)
print(cm)