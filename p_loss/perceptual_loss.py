import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms


# 定义感知损失类
class PerceptualLoss(nn.Module):
    def __init__(self, layers=['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']):
        super(PerceptualLoss, self).__init__()
        # 加载预训练的 VGG16 模型
        vgg16 = models.vgg16(pretrained=True).features
        self.layers = layers
        self.vgg_layers = nn.ModuleList(vgg16).eval()  # 提取特定层

        # 冻结 VGG16 权重，不进行反向传播
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.layer_indices = self.get_layer_indices()

    def get_layer_indices(self):
        # 定义每个层的索引
        mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }
        return [mapping[layer] for layer in self.layers]

    def forward(self, x, y):
        # 将输入图像传入 VGG16 提取特征
        x_features = self.extract_features(x)
        y_features = self.extract_features(y)

        # 计算每层的感知损失（L1 损失或 L2 损失）
        loss = 0
        for xf, yf in zip(x_features, y_features):
            loss += nn.functional.l1_loss(xf, yf)

        return loss

    def extract_features(self, x):
        features = []
        for idx, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if idx in self.layer_indices:
                features.append(x)
        return features


# 使用感知损失函数的示例
def calculate_perceptual_loss(gen_image, target_image):
    # 定义图像的标准化转换（与 VGG16 训练时保持一致）
    # transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # 将图像标准化
    # gen_image = transform(gen_image)
    # target_image = transform(target_image)

    # 初始化感知损失对象
    perceptual_loss_fn = PerceptualLoss()

    # 计算感知损失
    with torch.no_grad():
        loss = perceptual_loss_fn(gen_image, target_image)
    return loss
