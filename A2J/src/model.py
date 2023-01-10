import torch.nn as nn
import resnet
import torch


class DepthRegressionModel(nn.Module):
    def __init__(
        self, num_features_in, num_anchors=16, num_classes=15, feature_size=256
    ):
        super(DepthRegressionModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(
            feature_size, num_anchors * num_classes, kernel_size=3, padding=1
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.output(x)

        return x.view(x.shape[0], self.num_classes, 1, -1)  # B x C x 1 x (AxHxW)


class RegressionModel(nn.Module):
    def __init__(
        self, num_features_in, num_anchors=16, num_classes=15, feature_size=256
    ):
        super(RegressionModel, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(
            feature_size, num_anchors * num_classes * 2, kernel_size=3, padding=1
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.output(x)

        return x.view(x.shape[0], self.num_classes, 2, -1)  # B x C x 2 x (AxHxW)


class ClassificationModel(nn.Module):
    def __init__(
        self, num_features_in, num_anchors=16, num_classes=15, feature_size=256
    ):
        super(ClassificationModel, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU()
        self.output = nn.Conv2d(
            feature_size, num_anchors * num_classes, kernel_size=3, padding=1
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.act4(x)
        x = self.output(x)

        return x.view(x.shape[0], self.num_classes, 1, -1)  # B x C x 1 x (AxHxW)


class ResNetBackBone(nn.Module):
    def __init__(self):
        super(ResNetBackBone, self).__init__()

        self.model = resnet.resnet34(pretrained=True)

    def forward(self, x):
        # n, h, w = x.size()  # x: [B, 1, H ,W]
        # x = x.unsqueeze(1)  # depth
        # x = x.expand(n, 3, h, w)
        x = torch.stack((x,x,x), dim=1)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)  # [B, 256, H/16 ,W/16]

        return x


class A2J_model(nn.Module):
    def __init__(self, num_classes):
        super(A2J_model, self).__init__()
        self.Backbone = ResNetBackBone()  # 1 channel depth only, resnet50
        self.regressionModel = RegressionModel(
            num_features_in=256, num_classes=num_classes
        )
        self.classificationModel = ClassificationModel(
            num_features_in=256, num_classes=num_classes
        )
        self.softmax = nn.Softmax(dim=-1)
        self.anchors = self._set_anchors()

    def forward(self, x):
        x = self.Backbone(x)
        classification = self.classificationModel(x)
        regression = self.regressionModel(x)
        reg = self.anchors + regression
        reg_weight = self.softmax(reg)
        dt_keypoints = (reg_weight * reg).sum(dim=-1)
        return dt_keypoints

    def _set_anchors(self):
        # input_size = 256

        anchors = torch.Tensor(
            [
                [16 * x + 4 * i + 2, 16 * y + 4 * j + 2]
                for j in range(4)
                for i in range(4)
                for y in range(16)
                for x in range(16)
            ]
        )

        return anchors.permute(1, 0).contiguous().view(1, 1, 2, -1).cuda().float()
