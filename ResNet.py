import torch
from torch import nn

#残差块
class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1):
        super(ResidualBlock, self).__init__()
        #第一次卷积需要下采样并将输出通道数翻倍（通过stride调整）。当需要下采样时，stride=2， output_channel=input_channel*2
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3,stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU(inplace=True)
        #之后的残差块中的卷积就不需要下采样了，不改变尺寸。kernel_size,stride,padding=(3,1,1)时尺寸不变，=(3,2,1)时尺寸减半。
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(output_channel)

        self.shortcut = nn.Sequential()
        if stride !=1 or input_channel != output_channel:
            #用1*1的卷积核将输入特征的尺寸和通道数与输出特征的尺寸和通道数对其，这样才可以保证输入可以和输出相加
            self.shortcut = nn.Sequential(nn.Conv2d(input_channel,output_channel,kernel_size=1,stride=stride),
                                          nn.BatchNorm2d(output_channel))


    def forward(self,x):
        #对输入特征进行特征对齐处理
        identity = x
        #残差块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        #这一步就是输入与输出相加。
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

#resnet
class ResNet18(nn.Module):
    def __init__(self,num_classes=1000):
        super(ResNet18, self).__init__()
        #先进行一个7*7的卷积
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #第一个残差块不改变图像的尺寸，所以stride=1
        self.layer1 = self._make_layer_(64,64,1,stride=1)
        #第二个残差块需要进行一个特征图缩小并提升维度，所以stride=2
        self.layer2 = self._make_layer_(64,128,2,stride=2)
        #之后的残差块都需要缩小尺寸，提升维度
        self.layer3 = self._make_layer_(128,256,2,stride=2)
        #同上
        self.layer4 = self._make_layer_(256,512,2,stride=2)
        #一个总的平均池化加全连接层
        self.avpool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512,num_classes)#几分类num_classes就等于几


    def _make_layer_(self,input_channel,output_channel,blocks,stride=1):
        layer = []
        layer.append(ResidualBlock(input_channel,output_channel,stride))
        for _ in range(1,blocks):#遍历次数取决于残差块数
            layer.append(ResidualBlock(output_channel,output_channel))
        return nn.Sequential(*layer)

    def forward(self,x):
        #初始卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.maxpool(out)
        #残差块
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avpool(out)
        out = torch.flatten(out,1)
        out = self.linear(out)
        return out

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet18()
from torchsummary import summary
summary(model,(3,224,224),device='cpu')