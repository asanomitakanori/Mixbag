import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

from losses import PartialProportionLoss

class Attention(nn.Module):
    def __init__(self, w_n=0.01, w_p=10, w_MIL=1):
        super().__init__()
        self.num_classes = 3

        self.resnet = resnet18(pretrained=True)
        # self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Sequential()

        self.classifier_posinega = nn.Sequential(
            nn.Linear(512, 2),
        )

        self.classifier_class = nn.Sequential(
            nn.Linear(512, self.num_classes),
        )

        self.w_n ,self.w_p, self.w_MIL = w_n, w_p, w_MIL
        self.CELoss = nn.CrossEntropyLoss()
        self.PPLoss = PartialProportionLoss(self.w_n ,self.w_p)

    def forward(self, x):
        (N, B, C, W, H) = x.size()
        x = x.reshape(-1, C, W, H)

        x = self.feature_extraction(x) 
        
        x = x.reshape(N, B, -1)
        
        y_MIL = self.MIL_part(x)
        y, y_LLP = self.LLP_part(x)
        
        return y, y_MIL, y_LLP

    def feature_extraction(self, x):
        return self.resnet(x)

    def MIL_part(self, x):
        x = x.mean(1)
        y = self.classifier_posinega(x)
        return y
        
    def LLP_part(self, x):
        y = self.classifier_class(x)
        y = F.softmax(y, dim=-1)
        y_LLP = y.mean(-2)
        return y, y_LLP


    def calculate_loss_MIL(self, y_MIL, bag_label):
        return self.CELoss(y_MIL, bag_label)

    def calculate_loss_LLP(self, y_LLP, lp):
        return self.PPLoss(y_LLP, lp)

    def calculate_loss(self, y_MIL, bag_label, y_LLP, lp):
        loss_MIL = self.calculate_loss_MIL(y_MIL, bag_label)
        loss_LLP = self.calculate_loss_LLP(y_LLP, lp)[0]

        loss = loss_LLP + self.w_MIL * loss_MIL
        
        return loss

# class Attention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.num_classes = 3

#         self.resnet = resnet18(weights=None)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)
        

#         self.classifier_posinega = nn.Sequential(
#             nn.ReLU(),
#             nn.Linear(self.num_classes, 2),
#         )

#         self.CELoss = nn.CrossEntropyLoss()
#         self.PPLoss = PartialProportionLoss()

#     def forward(self, x):
#         (N, B, C, W, H) = x.size()
#         x = x.reshape(-1, C, W, H)
#         y = self.feature_extraction(x) 
#         y = y.reshape(N, B, -1)
        
#         pred_lp = F.softmax(y, dim=-1).mean(-2)
#         pred_pn = self.classifier_posinega(y.mean(-2))

#         return y, pred_pn, pred_lp

#     def MIL_part(self, x):
#         x = x.mean(1)
#         y = self.classifier_posinega(x)
#         return y

#     def feature_extraction(self, x):
#         return self.resnet(x)

#     def calculate_loss_MIL(self, y_MIL, bag_label):
#         return self.CELoss(y_MIL, bag_label)

#     def calculate_loss_LLP(self, prop_lp, lp):
#         return self.PPLoss(prop_lp, lp)

#     def calculate_loss(self, y_MIL, bag_label, prop_lp, lp):
#         loss_MIL = self.calculate_loss_MIL(y_MIL, bag_label)
#         loss_LLP = self.calculate_loss_LLP(prop_lp, lp)[0]

#         loss = loss_LLP + 0.01*loss_MIL
        
#         return loss



# class Attention(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.num_classes = 3

#         self.resnet = resnet18(weights=None)
#         self.resnet.fc = nn.Linear(self.resnet.fc.in_features, self.num_classes)

#         self.CELoss = nn.CrossEntropyLoss()
#         self.PPLoss = PartialProportionLoss()

#     def forward(self, x):
#         (N, B, C, W, H) = x.size()
#         x = x.reshape(-1, C, W, H)
#         y = self.feature_extraction(x) 
#         y = y.reshape(N, B, -1)
        
#         y = F.softmax(y, dim=-1)
#         pred_lp = y.mean(-2)
        
#         pred_pn = torch.concat([pred_lp[:, :1], pred_lp[:, 1:].sum(-1, keepdim=True)], dim=1)
        
#         return y, pred_pn, pred_lp


#     def feature_extraction(self, x):
#         return self.resnet(x)

#     def calculate_loss_MIL(self, y_MIL, bag_label):
#         return self.CELoss(y_MIL, bag_label)

#     def calculate_loss_LLP(self, prop_lp, lp):
#         return self.PPLoss(prop_lp, lp)

#     def calculate_loss(self, y_MIL, bag_label, prop_lp, lp):
#         loss_MIL = self.calculate_loss_MIL(y_MIL, bag_label)
#         loss_LLP = self.calculate_loss_LLP(prop_lp, lp)[0]

#         loss = loss_LLP + 100*loss_MIL
        
#         return loss