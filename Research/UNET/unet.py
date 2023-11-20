import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

'''
implement double convolution in as class 
just to save from code repetition
'''
class DConv(nn.Module):
    def __init__(self, ichans, ochans):
        super(DConv, self).__init__()
        self.conv = nn.Sequential(
            # Conv1
            nn.Conv2d(ichans, ochans, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ochans),
            nn.ReLU(inplace=True),
            # Conv2
            nn.Conv2d(ochans, ochans, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ochans),
            nn.ReLU(inplace=True)
        )
    def forward(self, A):
        return self.conv(A)
    
'''
UNET Class to implement that main unet architecture
'''    
class Unet(nn.Module):
    def __init__(self, ichans = 3, ochans=1, features=[64, 128, 256, 512]):
        super(Unet, self).__init__()
        self.ulayers = nn.ModuleList()
        self.dlayers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 

        # down unet layers
        for feature in features:    
            self.dlayers.append(DConv(ichans, feature))
            ichans = feature

        # up unet layers
        for feture in reversed(features):
            self.ulayers.append(
                nn.ConvTranspose2d(
                    feture * 2, feature, kernel_size=2 , stride= 2
                )
            )
            self.ulayers.append(DConv(feature * 2, feature))

        self.bottom = DConv(features[-1], features[-1] * 2)    
        self.final = nn.Conv2d(features[0], ochans, kernel_size=1 )

    def forward(self, a):
            skips = []

            for dn in self.dlayers:
                 a = dn(a)
                 skips.append(a)
                 a = self.pool(a)

            a = self.bottom(a)

            skips = skips[::-1]

            for idx in range(0, len(self.ulayers), 2):
                 a = self.ulayers[idx](a)
                 skips = skips[idx//2]

                 if a.shape != skips.shape:
                      a = TF.resize(a, size=skips.shape[2:])
                 
                 print(str(a.shape)  + ' : ' + str(skips.shape))
                 concat_skip = torch.cat((skips , a), dim=1)
                 a = self.ulayers[idx+1](concat_skip)

            return self.final(a)     

# define sample very basic test just to verify
def test():
     z = torch.randn((3, 1, 160, 160))
     model = Unet(ichans=1, ochans=1)
     preds = model(z)

     print(z.shape)
     print(preds.shape)    

     assert preds.shape == z.shape

# test the model architecture
if __name__=='__main__':
    test()     
