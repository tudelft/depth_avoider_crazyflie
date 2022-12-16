
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import nemo
from torchsummary import summary

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')         
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.bypass = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn_bypass = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu3 = nn.ReLU(inplace=False)
        # self.add = nemo.quant.pact.PACT_IntegerAdd()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x_bypass = self.bypass(identity)
        x_bypass = self.bn_bypass(x_bypass) # added DEBUG
        x_bypass = self.relu3(x_bypass) # added DEBUG
        # x += self.relu3(x_bypass)
        # x = self.add(x, x_bypass)
        x = x + x_bypass
        return x


class ResBlockEq(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockEq, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros')         
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        # self.add = nemo.quant.pact.PACT_IntegerAdd()

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        # x = self.add(x, identity)
        x = x + identity
        return x

class ConvAddBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvAddBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='reflect')
        self.relu = nn.ReLU6(inplace=False) # BE CAREFUL! was nn.ReLU 
        # self.minus_one = - torch.ones((64, 28, 40), dtype=torch.int8).to('cuda')

    def forward(self, x):
        # x = self.upconv_4_1_pad(x)
        x = self.conv(x)
        # x = x + 1.0
        x = self.relu(x)
        # x = self.add(x, -torch.ones_like(x)) # x = x + (-1.0)
        # x = self.add(x, self.minus_one)
        return x
    

class ConvSigBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSigBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='reflect')
        # self.bn = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # self.sigmoid = nn.Sigmoid()
        self.relu6 = nn.ReLU6(inplace=False) # fake sigmoid

    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        # x = self.sigmoid(x)
        x = self.relu6(x)
        return x


class nanoDepth_relu6(nn.Module):
    def __init__(self):
        super(nanoDepth_relu6, self).__init__()
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_0 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False)

        self.resBlock1 = ResBlockEq(16, 16)
        self.resBlock2 = ResBlock(16, 32, 2)
        self.resBlock3 = ResBlock(32, 64, 1)
        self.resBlock4 = ResBlock(64, 128, 1)

        self.decoderBlock1 = ConvAddBlock(128, 64)
        self.decoderBlock2 = ConvAddBlock(128, 1) # 64

        # self.dispBlock = ConvSigBlock(64, 1) # not okay for onboard but many models already trained

        fc_size = 1*40*28
        self.fc = nn.Linear(in_features=fc_size, out_features=8, bias=False)

        # # NOTE this is parameter initialization, can be commented out
        # CANNOT use this, will lead to all zero output!!!
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
                
        for name, param in self.decoderBlock2.named_parameters():
            # print(name, param.size())
            # if 'weight' in name:
            #     param.data.fill_(1.0)
            if 'bias' in name:
                # print(param.data)
                param.data.fill_(0.02)

        
        print("NanoDepth_relu66666666666666!")
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(num_params)


        # self.load_state_dict(torch.load("/home/adr/datasets/crazyfly/nanodepth/PoseDepth1101013919corrider_relu6_consist_0_10/nanodepth_relu6.pth"))
        # print("Loading pretrained nanoDepth_relu6")

        # fc_mtrx_sub = torch.zeros(8, 40)
        # for i in range (8):
        #     fc_mtrx_sub[i, (i*5):((i+1)*5)] = 1.0 # TODO we may only want the middle rows
        
        # fc_mtrx = fc_mtrx_sub
        # for i in range (28-1):
        #     fc_mtrx = torch.cat((fc_mtrx, fc_mtrx_sub), dim=1)
        # # print(fc_mtrx.size())
        # fc_mtrx = fc_mtrx / (5*28)

        # with torch.no_grad():
        #     for name, param in self.fc.named_parameters():
        #         if 'weight' in name:
        #             param.copy_(fc_mtrx)

        # to_save = self.state_dict()
        # torch.save(to_save, "checkpoint/nanodepth_relu6_140.pth", _use_new_zipfile_serialization=False)

        # self.models["nanodepth"].load_state_dict(torch.load("/home/adr/datasets/crazyfly/nanodepth/Depth1108151813corrider_KD/models/weights_lastest/nanodepth.pth", map_location=self.device))
        # print("Loading pretrained nanoDepth_fc")

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu_0(x)
        x = self.maxpool(x)
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        out3 = self.resBlock3(x)
        x = self.resBlock4(out3)
        x = self.decoderBlock1(x)
        x = torch.cat((x, out3), 1)
        disp = self.decoderBlock2(x)
        # disp = self.dispBlock(x) # [28, 40]

        x = disp.flatten(1)
        x = self.fc(x)

        # print("disp mean: ")
        # print(disp[0, 0, :, :5].mean())
        # print(disp[0, 0, :, 5:10].mean())
        # print(disp[0, 0, :, 10:15].mean())
        # print(disp[0, 0, :, 15:20].mean())
        # print(disp[0, 0, :, 20:25].mean())
        # print(disp[0, 0, :, 25:30].mean())
        # print(disp[0, 0, :, 30:35].mean())
        # print(disp[0, 0, :, 35:].mean())
        # print("  ")
        # print("fc output: ")
        # print(x)
        # print("  ")

        return disp/6.0 # for training

class nanoDepth_fc(nn.Module):
    def __init__(self):
        super(nanoDepth_fc, self).__init__()
        self.first_conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3, dilation=1, groups=1, bias=False, padding_mode='zeros')
        self.bn1 = nn.BatchNorm2d(num_features=16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu_0 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, return_indices=False, ceil_mode=False)

        self.resBlock1 = ResBlockEq(16, 16)
        self.resBlock2 = ResBlock(16, 32, 2)
        self.resBlock3 = ResBlock(32, 64, 2) # [20, 14]
        self.resBlock4 = ResBlock(64, 128, 1)

        self.decoderBlock1 = ConvAddBlock(128, 64)
        self.decoderBlock2 = ConvAddBlock(128, 32) 

        fc_size = 32*20*14
        self.fc = nn.Linear(in_features=fc_size, out_features=8, bias=False)

        # NOTE this is parameter initialization, can be commented out
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu_0(x)
        
        x = self.maxpool(x)
        x = self.resBlock1(x)
        
        x = self.resBlock2(x)
        out3 = self.resBlock3(x)
        
        x = self.resBlock4(out3)
        
        x = self.decoderBlock1(x)

        x = torch.cat((x, out3), 1)
        x = self.decoderBlock2(x)
        # print(x[0, 0, 0, :10])
        x = x.flatten(1)
        x = self.fc(x)
        return x

# if __name__ == '__main__':
#     model = nanoDepth_relu6().to('cuda')
#     summary(model, (1, 224, 320))
#     # summary(model, (1, 112, 160))