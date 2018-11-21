import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import vgg19
import math

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

        vgg19_model = vgg19(pretrained=True)

        # Extracts features at the 11th layer
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:12])

    def forward(self, img):
        out = self.feature_extractor(img)
        return out

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=False, upsample=False, nobn = False):
        super(BasicBlock, self).__init__()
        self.upsample = upsample
        self.downsample = downsample
        self.nobn = nobn
        if self.upsample:
            self.conv1 = nn.ConvTranspose2d(inplanes, planes, 4, 2, 1)
        else:
            self.conv1 = conv3x3(inplanes, planes, stride)
        if not self.nobn:
            self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.downsample:
            self.conv2 =nn.Sequential(nn.AvgPool2d(2,2), conv3x3(planes, planes))
        else:
            self.conv2 = conv3x3(planes, planes)
        if not self.nobn:
            self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes or self.upsample or self.downsample:
            if self.upsample:
                self.skip = nn.ConvTranspose2d(inplanes, planes, 4, 2, 1)
            elif self.downsample:
                self.skip = nn.Sequential(nn.AvgPool2d(2,2), nn.Conv2d(inplanes, planes, 1, 1))
            else:
                self.skip = nn.Conv2d(inplanes, planes, 1, 1, 0)
        else:
            self.skip = None
        self.stride = stride

    def forward(self, x):
        residual = x
        if not self.nobn:
            out = self.bn1(x)
            out = self.relu(out)
        else:
            out = self.relu(x)
        out = self.conv1(out)
        if not self.nobn:
            out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.skip is not None:
            residual = self.skip(x)
        out += residual
        return out


class GeneratorResNet(nn.Module):
    def __init__(self, ngpu=1):
        super(GeneratorResNet, self).__init__()
        self.ngpu = ngpu
        res_units = [256, 128, 96]
        inp_res_units = [
            [256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
             256], [256, 128, 128], [128, 96, 96]]

        self.layers_set = []
        self.layers_set_up = []
        self.layers_set_final = nn.ModuleList()
        self.layers_set_final_up = nn.ModuleList()

        self.a1 = nn.Sequential(nn.Conv2d(256, 128, 1, 1))
        self.a2 = nn.Sequential(nn.Conv2d(128, 96, 1, 1))

        self.layers_in = conv3x3(3, 256)

        layers = []
        for ru in range(len(res_units) - 1):
            nunits = res_units[ru]
            curr_inp_resu = inp_res_units[ru]
            self.layers_set.insert(ru, [])
            self.layers_set_up.insert(ru, [])

            if ru == 0:
                num_blocks_level = 12
            else:
                num_blocks_level = 3

            for j in range(num_blocks_level):
                # if curr_inp_resu[j]==3:
                self.layers_set[ru].append(BasicBlock(curr_inp_resu[j], nunits))
                # else:
                # layers.append(MyBlock(curr_inp_resu[j], nunits))

            self.layers_set_up[ru].append(nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True))

            self.layers_set_up[ru].append(nn.BatchNorm2d(nunits))
            self.layers_set_up[ru].append(nn.ReLU(True))
            self.layers_set_up[ru].append(nn.ConvTranspose2d(nunits, nunits, kernel_size=1, stride=1))
            self.layers_set_final.append(nn.Sequential(*self.layers_set[ru]))
            self.layers_set_final_up.append(nn.Sequential(*self.layers_set_up[ru]))

        nunits = res_units[-1]
        layers.append(conv3x3(inp_res_units[-1][0], nunits))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(inp_res_units[-1][1], nunits, kernel_size=1, stride=1))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(nunits, 3, kernel_size=1, stride=1))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers_in(input)
        for ru in range(len(self.layers_set_final)):
            if ru == 0:
                temp = self.layers_set_final[ru](x)
                x = x + temp
            elif ru == 1:
                temp = self.layers_set_final[ru](x)
                temp2 = self.a1(x)
                x = temp + temp2
            elif ru == 2:
                temp = self.layers_set_final[ru](x)
                temp2 = self.a2(x)
                x = temp + temp2
            x = self.layers_set_final_up[ru](x)

        x = self.main(x)

        return x


class GeneratorResNet_2(nn.Module):
    def __init__(self, ngpu=1):
        super(GeneratorResNet_2, self).__init__()
        self.ngpu = ngpu
        res_units = [64, 128, 256, 512, 256, 128]
        sampling_units = [-1,-1,-1,-1,1,1]
        inp_res_units = [
            [64, 64], [64, 128], [128, 256], [256, 512], [512, 256], [256, 128]
            ]

        self.layers_set = []
        self.layers_set_up = []
        self.layers_set_final = nn.ModuleList()
        self.layers_set_final_up = nn.ModuleList()

        self.layers_in = conv3x3(3, 64)

        layers = []
        for ru in range(len(res_units) - 1):
            nunits = res_units[ru]
            curr_inp_resu = inp_res_units[ru]
            self.layers_set.insert(ru, [])
            self.layers_set_up.insert(ru, [])


            num_blocks_level = 2


            for j in range(num_blocks_level):
                # if curr_inp_resu[j]==3:
                if j==0 :
                    if sampling_units[ru] == -1:
                        self.layers_set[ru].append(BasicBlock(curr_inp_resu[j], nunits, downsample=True))
                    elif sampling_units[ru] == 1:
                        self.layers_set[ru].append(BasicBlock(curr_inp_resu[j], nunits, upsample=True))
                else :
                    self.layers_set[ru].append(BasicBlock(curr_inp_resu[j], nunits))
                # else:
                # layers.append(MyBlock(curr_inp_resu[j], nunits))

            self.layers_set_final.append(nn.Sequential(*self.layers_set[ru]))

        nunits = res_units[-1]
        layers.append(conv3x3(inp_res_units[-1][0], nunits))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(inp_res_units[-1][1], nunits, kernel_size=1, stride=1))
        layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(nunits, 3, kernel_size=1, stride=1))
        layers.append(nn.Tanh())

        self.main = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers_in(input)
        for ru in range(len(self.layers_set_final)):
            if ru == 0:
                temp = self.layers_set_final[ru](x)
                x = x + temp
            elif ru == 1:
                temp = self.layers_set_final[ru](x)
                temp2 = self.a1(x)
                x = temp + temp2
            elif ru == 2:
                temp = self.layers_set_final[ru](x)
                temp2 = self.a2(x)
                x = temp + temp2
            x = self.layers_set_final_up[ru](x)

        x = self.main(x)

        return x

# class GeneratorResNet(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=16):
#         super(GeneratorResNet, self).__init__()

#         # First layer
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels, 64, 9, 1, 4),
#             nn.ReLU(inplace=True)
#         )

#         # Residual blocks
#         res_blocks = []
#         for _ in range(n_residual_blocks):
#             res_blocks.append(ResidualBlock(64))
#         self.res_blocks = nn.Sequential(*res_blocks)

#         # Second conv layer post residual blocks
#         self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64))

#         # Upsampling layers
#         upsampling = []
#         for out_features in range(2):
#             upsampling += [ nn.Conv2d(64, 256, 3, 1, 1),
#                             nn.BatchNorm2d(256),
#                             nn.PixelShuffle(upscale_factor=2),
#                             nn.ReLU(inplace=True)]
#         self.upsampling = nn.Sequential(*upsampling)

#         # Final output layer
#         self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, 9, 1, 4), nn.Tanh())

#     def forward(self, x):
#         out1 = self.conv1(x)
#         out = self.res_blocks(out1)
#         out2 = self.conv2(out)
#         out = torch.add(out1, out2)
#         out = self.upsampling(out)
#         out = self.conv3(out)
#         return out

# class Discriminator(nn.Module):
#     def __init__(self, in_channels=3):
#         super(Discriminator, self).__init__()

# #         def discriminator_block(in_filters, out_filters, stride, normalize):
# #             """Returns layers of each discriminator block"""
# #             layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
# #             if normalize:
# #                 layers.append(nn.BatchNorm2d(out_filters))
# #             layers.append(nn.LeakyReLU(0.2, inplace=True))
# #             return layers


#         self.layers_set = []

#         self.layers_set_final = nn.ModuleList()

#         #self.maxpool = nn.Sequential(nn.MaxPool2d(2,2), nn.MaxPool2d(2,2))

#         res_units = [16, 32, 64, 128, 256, 512]

#         layers = []
#         in_filters = in_channels

#         for ru in range(len(res_units) - 1):
#             out_filters = res_units[ru]

#             self.layers_set.insert(ru, [])
#             if ru == 0 or ru == 1 :
#                 self.layers_set[ru].append(BasicBlock(in_filters, out_filters, 1, True, False, True))
#             else :
#                 self.layers_set[ru].append(BasicBlock(in_filters, out_filters, 1, False, False, True))



#             in_filters = out_filters

#             self.layers_set_final.append(nn.Sequential(*self.layers_set[ru]))


#         # Output layer
#         layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

#         self.model = nn.Sequential(*layers)

#     def forward(self, img):
#         x = self.layers_set_final(img)
#         return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for out_filters, stride, normalize in [ (64, 1, False),
                                                (64, 2, True),
                                                (128, 1, True),
                                                (128, 2, True),
                                                (256, 1, True),
                                                (256, 2, True),
                                                (512, 1, True),
                                                (512, 2, True),]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        # Output layer
        layers.append(nn.Conv2d(out_filters, 1, 3, 1, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)
