import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet


def main():
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),#因为1是从网上下载的，它不一定是32*32，这一步就是把他转成32*32
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet()
    net.load_state_dict(torch.load('Lenet.pth'))

    im = Image.open('1.jpg')
    im = transform(im)  # [C, H, W]
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W]   就是要从根据网络才开始怎么输入，补齐他的参数，然后走完整个网络，得到输出。  这个函数的意思就是增加一个新的维度

    with torch.no_grad():
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy()   #这里输出的，第一个维度是bantch，第二个才是
    print(classes[predict.item()])  # .item() 从单元素数组中提取Python标量


if __name__ == '__main__':
    main()
