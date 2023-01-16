

#encoding:utf-8
import os
import time
import sys
import logging
import warnings
import argparse
from pylab import *                                 #支持中文
from PIL import Image
import matplotlib.pyplot as plt
import h5py
import torchvision
from torchvision import transforms
import torchvision.models as models
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
import AveragePrecision
from RandAugment import RandAugment
from Module import EfficientNet
from ranger import Ranger  # this is from ranger.py
from ranger import RangerVA  # this is from ranger913A.py
from ranger import RangerQH  # this is from rangerqh.py
import AveragePrecision_vec
# 常用参数
num_epochs = 1000
batch_size = 32
step_size = 10
classes = 32
INPUT_UNIT =1536
OUTPUT_UNIT = 400*32
feature_dim = 400*2


parser = argparse.ArgumentParser(description="Your project")  # 创建一个解析对象
# parser.add_argument("-f","--feature_dim",type = int, default = 400*2)  
# parser.add_argument("-a","--alpha",type = float, default = 1)  
# parser.add_argument("-r","--relation_dim",type = int, default = 8)
# parser.add_argument("-b","--batch_size",type = int, default = 32)
# parser.add_argument("-e","--episode",type = int, default= 150000)
parser.add_argument("-lr","--learning_rate", type = float, default = 0.0001)
parser.add_argument("-g", "--gpu_devices", type=int, nargs='+', default=0)
# parser.add_argument("-u","--hidden_unit",type=int,default=300)
# parser.add_argument("-i", "--input_unit", type=int, default=1536)
# parser.add_argument("-o", "--output_unit", type=int, default=400 * classes)
args = parser.parse_args()  # 解析参数

gpu_devices = ','.join([str(id) for id in args.gpu_devices])
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LEARNING_RATE = args.learning_rate

# 忽略警告
warnings.filterwarnings("ignore")
#打印日志
# class Logger(object):
#     def __init__(self, filename="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")

#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)

#     def flush(self):
#         pass
# sys.stdout = Logger('/home/wangzeyu/code/pytorch/efficientnet_pytorch/data/relation_'+str(LEARNING_RATE)+'_'+str(alpha)+'.txt')

# 定义数据读入
def Load_Image_Information(path):
    # 图像存储路径
    image_Root_Dir = r'/home/wangzeyu/data/python/multi_label'  # 数据集图像路径
    # 获取图像的路径
    iamge_Dir = os.path.join(image_Root_Dir, path)
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    return Image.open(iamge_Dir).convert('RGB')

def Load_Image_Information1(path):
    # path
    image_Root_Dir = r'/home/wangzeyu/data/R/multi_label'
    iamge_Dir = os.path.join(image_Root_Dir, path)
    return Image.open(iamge_Dir).convert('RGB')

class my_Data_Set4(nn.Module):
    def __init__(self, txt,txt1, transform=None, target_transform=None, loader=None):
        super(my_Data_Set4, self).__init__()
        fp = open(txt, 'r')
        images = []
        labels = []
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels.append([float(l) for l in information[1:len(information)]])

        fp1 = open(txt1, 'r')
        images1 = []
        labels1 = []
        for line in fp1:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images1.append(information[0])
            labels1.append([float(l) for l in information[1:len(information)]])

        self.images = images+images1
        self.labels = labels+labels1
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.images1=[]
        self.labels1=[]
        self.judge=[]
        num=0
        for i in range(0,len(self.images)):
            compare=[]
            compare1=[]
            new=[]
            label=self.labels[i]
            for j in range(0,32):
                compare.append(0)
            for j in range(0,13):
                new.append(0.0)
            for j in range(0,13):
                compare1.append(0)
            if(label[30]==1.0):
                label[30]=0.0
                new[12]=1.0
            if(label[9]==1.0):
                label[9]=0.0
                new[4]=1.0
            if(label[10]==1.0):
                label[10]=0.0
                new[4]=1.0  
            if(label==compare):
                num=num+1
                self.images1.append(self.images[i]) 
                self.labels1.append(new)
                self.judge.append(1)
    def __getitem__(self, item):
        imageName = self.images[item]
        label = self.labels[item]
        image = self.loader(imageName)
        if self.transform is not None:
            image = self.transform(image)
        return image, imageName

    def __len__(self):
        return len(self.images)

class my_Data_Set3(nn.Module):
    def __init__(self, txt,transform=None, target_transform=None, loader=None):
        super(my_Data_Set3, self).__init__()
        fp = open(txt, 'r')
        images = []
        labels = []
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels.append([float(l) for l in information[1:len(information)]])

        
        
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.images1=[]
        self.labels1=[]
        self.judge=[]
        num=0
        for i in range(0,len(self.images)):
            compare=[]
            compare1=[]
            new=[]
            label=self.labels[i]
            for j in range(0,13):
                compare.append(0)
            for j in range(0,3):
                new.append(0.0)
            for j in range(0,32):
                compare1.append(0)
            if(label[0]==1.0):
                new[0]=1.0

            if(label[1]==1.0):
                new[0]=1.0

            if(label[2]==1.0):
                new[0]=1.0

            if(label[3]==1.0):
                new[0]=1.0

            if(label[4]==1.0):
                new[0]=1.0

            if(label[5]==1.0):
                new[1]=1.0

            if(label[6]==1.0):
                new[2]=1.0

            if(label[7]==1.0):
                new[2]=1.0

            if(label[8]==1.0):
                new[2]=1.0


            if(label[9]==1.0):
                new[2]=1.0

            if(label[10]==1.0):
                new[2]=1.0

            if(label[11]==1.0):
                new[2]=1.0

            if(label[12]==1.0):
                new[0]=1.0

            if(label[13]==1.0):
                new[0]=1.0

            if(label[14]==1.0):
                new[2]=1.0


            if(label[15]==1.0):
                new[2]=1.0

            if(label[16]==1.0):
                new[2]=1.0

            if(label[17]==1.0):
                new[0]=1.0

            if(label[18]==1.0):
                new[2]=1.0


            if(label[19]==1.0):
                new[2]=1.0

            if(label[20]==1.0):
                new[2]=1.0

            if(label[21]==1.0):
                new[0]=1.0

            if(label[22]==1.0):
                new[2]=1.0

            if(label[23]==1.0):
                new[0]=1.0

            if(label[24]==1.0):
                new[2]=1.0

            if(label[25]==1.0):
                new[2]=1.0

            if(label[26]==1.0):
                new[0]=1.0

            if(label[27]==1.0):
                new[0]=1.0

            if(label[28]==1.0):
                new[2]=1.0

            if(label[29]==1.0):
                new[2]=1.0


            if(label[30]==1.0):
                new[0]=1.0

            if(label[31]==1.0):
                new[0]=1.0
           
            if(1==1):
                self.images1.append(self.images[i]) 
                self.labels1.append(new)
    def __getitem__(self, item):
        imageName = self.images[item]
        label = self.labels1[item]
        image = self.loader(imageName)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.FloatTensor(label)
        return image, label

    def __len__(self):
        return len(self.images)

class my_Data_Set2(nn.Module):
    def __init__(self, txt,transform=None, target_transform=None, loader=None):
        super(my_Data_Set2, self).__init__()
        fp = open(txt, 'r')
        images = []
        labels = []
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels.append([float(l) for l in information[1:len(information)]])

        

        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.images1=[]
        self.labels1=[]
        self.judge=[]
        num=0
        for i in range(0,len(self.images)):
            compare=[]
            compare1=[]
            new=[]
            label=self.labels[i]
            for j in range(0,13):
                compare.append(0)
            for j in range(0,3):
                new.append(0.0)
            for j in range(0,32):
                compare1.append(0)
            if(label[0]==1.0):
                new[0]=1.0

            if(label[1]==1.0):
                new[0]=1.0

            if(label[2]==1.0):
                new[0]=1.0

            if(label[3]==1.0):
                new[0]=1.0

            if(label[4]==1.0):
                new[2]=1.0

            if(label[5]==1.0):
                new[0]=1.0

            if(label[6]==1.0):
                new[1]=1.0

            if(label[7]==1.0):
                new[2]=1.0

            if(label[8]==1.0):
                new[1]=1.0
                new[2]=1.0

            if(label[9]==1.0):
                new[2]=1.0

            if(label[10]==1.0):
                new[0]=1.0

            if(label[11]==1.0):
                new[2]=1.0

            if(label[12]==1.0):
                new[0]=1.0
           
            if(1==1):
                self.images1.append(self.images[i]) 
                self.labels1.append(new)
    def __getitem__(self, item):
        imageName = self.images[item]
        label = self.labels1[item]
        image = self.loader(imageName)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.FloatTensor(label)
        return image, label

    def __len__(self):
        return len(self.images)
# 定义自己数据集的数据读入类
class my_Data_Set(nn.Module):
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        fp = open(txt, 'r')
        images = []
        labels = []
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            labels.append([float(l) for l in information[1:len(information)]])
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = Load_Image_Information
        self.loader1=Load_Image_Information1
        self.judge=[]
        for i in range(0,len(self.images)):
            self.judge.append(0)
    def __getitem__(self, item):
        imageName = self.images[item]
        label = self.labels[item]
        if(self.judge[item]==0):
            image = self.loader(imageName)
        if (self.judge[item] == 1):
            image = self.loader1(imageName)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.FloatTensor(label)
        return image, label

    def __len__(self):
        return len(self.images)


# 加载数据
data_transforms = {
    "train": transforms.Compose([
        transforms.Scale((300, 300), 2),  # this is a list of PIL Images
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.892, 0.894, 0.894], std=[0.207, 0.188, 0.196]),
        transforms.RandomErasing()
    ]),
    "val": transforms.Compose([transforms.Scale((300, 300), 2),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.892, 0.894, 0.894], std=[0.207, 0.188, 0.196])
                               ])}
train_Data2 = my_Data_Set2(r'/home/wangzeyu/data/python/train_label.txt',
                           transform=data_transforms["val"],
                         loader=Load_Image_Information)
train_Data3 = my_Data_Set3(r'/home/wangzeyu/data/R/train_label.txt',
                           transform=data_transforms["val"],
                         loader=Load_Image_Information1)
train_Data4 = my_Data_Set4(r'/home/wangzeyu/data/R/train_label.txt',r'/home/wangzeyu/data/R/val_label.txt',
                           transform=data_transforms["val"],
                         loader=Load_Image_Information1)
train_Data = my_Data_Set(r'/home/wangzeyu/data/python/train_label.txt', transform=data_transforms["train"],
                         loader=Load_Image_Information)
train_Data.images=train_Data.images+train_Data4.images1
train_Data.labels=train_Data.labels+train_Data4.labels1
train_Data.judge=train_Data.judge+train_Data4.judge
train_Data.transform.transforms.insert(0, RandAugment(3, 10))
val_Data = my_Data_Set(r'/home/wangzeyu/data/python/val_label.txt', transform=data_transforms["val"],
                       loader=Load_Image_Information)
train_loader = torch.utils.data.DataLoader(train_Data, batch_size=batch_size, shuffle=True, num_workers=16)
train_loader3 = torch.utils.data.DataLoader(train_Data3, batch_size=batch_size, shuffle=True, num_workers=16)
train_loader2 = torch.utils.data.DataLoader(train_Data2, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = torch.utils.data.DataLoader(val_Data, batch_size=batch_size, shuffle=True, num_workers=16)
print("load dataset done")


# 进行迭代训练模型
epoch = 0
best_acc = 0.0
best_acc_vec = 0.0

class fully_connect(nn.Module):
    """docstring for ClassName"""

    def __init__(self, INPUT_UNIT, OUTPUT_UNIT):
        super(fully_connect, self).__init__()
        self.fc = nn.Linear(INPUT_UNIT, OUTPUT_UNIT)

    def forward(self, x):
        # out = F.leaky_relu(self.fc1(x))
        # out = F.sigmoid(self.fc2(out))
        out = F.relu(self.fc(x))
        return out

class RelationModule(torch.nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim):
        super(RelationModule, self).__init__()
        self.img_feature_dim = feature_dim
        self.classifier = self.fc_fusion()
    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 256
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck, 1),
                )
        return classifier
    def forward(self, input):
        input = input.view(input.size(0), self.img_feature_dim)
        input = self.classifier(input)
        return input
# 加载 ResNet 模型
'''
model = models.resnet50(pretrained=True).cuda()  
model.removed = list(model.children())[:-1]
model.fc = nn.Linear(2048, 3).cuda()    #全连接分类改为37类

model = models.vgg16(pretrained=True).cuda() 
model.classifier = nn.Sequential(nn.Linear(25088, 4096),      #vgg16
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 3)).cuda() 

for index, parma in enumerate(model.classifier.parameters()):
     if index == 6:
        parma.requires_grad = True
model = models.vgg16(pretrained=True).cuda() 
removed = list(model.classifier.children())[:-1]
model.classifier = torch.nn.Sequential(*removed)    #全连接分类改为37类
model.add_module('fc', torch.nn.Linear(4096, 3))
'''
model = models.resnet50(pretrained=True).cuda()
model=torch.load('resnet.pkl')
model.removed = list(model.children())[:-1]
model.fc = nn.Linear(2048, 3).cuda() 
optimizer =torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
print("ls= %.5f,bs= %2d" % (LEARNING_RATE, batch_size))


# save_path = './googleNet.pth'
def criterion(y_pred,y_true):
    y_pred = (1 - 2*y_true)*y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    ones = torch.ones_like(y_pred[...,:1])
    zeros= torch.zeros_like(y_pred[...,:1])
    y_pred_neg = torch.cat((16*(y_pred_neg+0.125),zeros),dim=-1)
    y_pred_pos = torch.cat((16*(y_pred_pos+0.125),zeros),dim=-1)
    #print( y_pred_neg)
    neg_loss = torch.logsumexp(y_pred_neg,dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos,dim=-1)
    return torch.mean(neg_loss + pos_loss)

matrix = np.zeros([13, 400])
index = 0
f = h5py.File('/home/wangzeyu/code/gcn/python_word2vec.h5', 'r')
for key in f.keys():
    matrix[index] = f[key].value
    index = index + 1
matrix = torch.from_numpy(matrix)





for epoch in range(1):
    # train
    model.train()
    batch_size_start = time.time()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader2):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
    
        outputs= model(images)   #前向传播求出预测的值
        criterion1 = nn.MultiLabelSoftMarginLoss().cuda()     #交叉熵
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    for i, (images, labels) in enumerate(train_loader3):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs= model(images)   #前向传播求出预测的值
        criterion1 = nn.MultiLabelSoftMarginLoss().cuda()     #交叉熵
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()



epoch = 0
best_acc = 0.0
best=0
'''
model.removed = list(model.children())[:-1]
model.fc = nn.Linear(4096, 13).cuda()    #全连接分类改为37类

model.classifier = nn.Sequential(nn.Linear(25088, 4096),      #vgg16
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.5),
                                 nn.Linear(4096, 13)).cuda() 

for index, parma in enumerate(model.classifier.parameters()):
     if index == 6:
        parma.requires_grad = True
'''
model.removed = list(model.children())[:-1]
model.fc = nn.Linear(2048, 13).cuda() 
optimizer =Ranger(model.parameters(), lr=LEARNING_RATE)
model.eval()
ap_meter = AveragePrecision.AveragePrecisionMeter(difficult_examples=False)
with torch.no_grad():
    for j, (inputs,labels) in enumerate(val_loader):
        batch_size_start = time.time()
        inputs = inputs.to(device)
        outputs= model(inputs)
        ap_meter.add(outputs.data, labels)
print(100 * ap_meter.value())
map = 100 * ap_meter.value().mean()
print(" Val BatchSize cost time :%.4f s" % (time.time() - batch_size_start))
print('Test Accuracy of the model on the 5000 Val images: %.4f' % (map))


