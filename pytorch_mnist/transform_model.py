# 将模型序列化导出

import torch
import cv2
import torch.nn.functional as F
from train import Net 
from torchvision import datasets, transforms
from PIL import Image
 
if __name__ == '__main__':
    device = torch.device('cpu')  
    model = torch.load('model.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式
 
    img = cv2.imread("img.jpg",0)  # 读取要预测的灰度图片
    img = Image.fromarray(img)
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
 
    img = trans(img)
    img = img.unsqueeze(0)  # 图片扩展多一维,[batch_size,通道,长，宽]
    img = img.to(device)
    traced_net = torch.jit.trace(model,img)
    traced_net.save("model.pt")

    print("模型序列化导出成功")