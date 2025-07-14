
import torch
import torch.nn as nn

# This dictionary represents the "genes" of this specific model.
# It's included for easy reference and reproducibility.
chromosome = {'conv1_filters': 64, 'conv1_kernel': 3, 'conv1_stride': 2, 'conv2_filters': 128, 'conv2_kernel': 3, 'conv3_filters': 192, 'conv4_filters': 256, 'conv5_filters': 256, 'fc1_neurons': 2048, 'fc2_neurons': 1024}

# The nn-dataset framework requires these functions.
def supported_hyperparameters():
    return {'lr', 'momentum', 'dropout'}

class Net(nn.Module):
    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        # Use the fixed learning rate and momentum for architecture search.
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])

    def learn(self, train_data):
        self.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criteria[0](outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.optimizer.step()

    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        in_channels = in_shape[0]

        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 4
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pooling makes the classifier robust to changes in the feature extractor's output size.
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # The input to the first Linear layer depends on the number of filters in the LAST conv layer.
        classifier_input_features = 256 * 6 * 6
        dropout_p = prm.get('dropout', 0.5) # Use dropout from prm, with a default

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(classifier_input_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_shape[0]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
