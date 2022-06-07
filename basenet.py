import torch.nn as nn
import torch
class EmbeddingNetHyperX(nn.Module):
    def __init__(self, input_channels, n_outputs=128, patch_size=5, n_classes=None):
        super(EmbeddingNetHyperX, self).__init__()
        self.dim = 200

        # 1st conv layer
        # input [input_channels x patch_size x patch_size]
        self.convnet = nn.Sequential(
            nn.Conv2d(input_channels, self.dim, kernel_size=1, padding=0),  # input channels
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(self.dim, self.dim, kernel_size=1, padding=0,),
            nn.BatchNorm2d(self.dim, momentum=1, affine=True,track_running_stats=True),
            nn.ReLU(),

            nn.AvgPool2d(patch_size, stride=1)

        )

        self.n_outputs = n_outputs
        self.fc = nn.Linear(self.dim, self.n_outputs)

    def extract_features(self, x):

        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc[0](output)

        return output

    def forward(self, x):

        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)

        return output

    def get_embedding(self, x):
        return self.forward(x)
    def output_num(self):
        return self.n_outputs

class ResClassifier(nn.Module):
    def __init__(self, num_classes=7,  num_unit=128, middle=64):
        super(ResClassifier, self).__init__()
        layers = []

        layers.append(nn.Linear(num_unit, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(middle, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Linear(middle, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):

        x = self.classifier(x)
        return x


