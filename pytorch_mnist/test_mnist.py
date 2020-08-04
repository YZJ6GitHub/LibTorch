# 单张图片测试

import torch
import cv2
import torch.nn.functional as F
from train import Net 
from torchvision import datasets, transforms
from PIL import Image
 
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式
 
    img = cv2.imread("img_3.jpg",0)  # 读取要预测的灰度图片
    img = Image.fromarray(img)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
 
    img = trans(img)
    img = img.unsqueeze(0)  # 图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    img = img.to(device)
    output = model(img)
    pred = output.max(1, keepdim=True)[1]
    pred = torch.squeeze(pred)
    print('检测结果为：%d' % (pred.cpu().numpy()))