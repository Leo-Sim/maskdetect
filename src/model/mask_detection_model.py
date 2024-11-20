import torch
from torch import optim, nn, utils, Tensor
from torch.nn import CrossEntropyLoss
from torchvision.transforms import transforms
import lightning as L
from torchmetrics.classification import F1Score
from torchmetrics import Accuracy, Precision, Recall
import torch.optim as optim


class MaskDetectionModel(L.LightningModule):
    NUM_OF_CLASSES = 43

    def __init__(self, num_of_classes, image_size=(128, 128), lr=0.0001, momentum=0.9):
        super().__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2, 1),
            #
        )

        # self.conv_layer = nn.Sequential(
        #     nn.Conv2d(3, 32, 3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 512, 3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )

    # No.2 trial
        # self.conv_layer = nn.Sequential(
        #     nn.Conv2d(3, 32, 3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, 3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )

        # self.conv_layer = nn.Sequential(
        #     nn.Conv2d(3, 16, 5, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(16, 32, 5, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2),
        #     nn.Conv2d(32, 64, 5, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        #     nn.MaxPool2d(2))

        # After passing nn.Sequential above, it will pass FC layer to determine the final label of the image
        conv_output_size = self._get_conv_output_size(image_size)

        self.linear_layer = nn.Sequential(
            nn.Linear(conv_output_size, num_of_classes),
            nn.Softmax(dim=1)
        )

        # self.linear_layer = nn.Sequential(
        #     nn.Dropout(0.5),
        #     nn.Linear(conv_output_size, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, num_of_classes)
        # )

    # No.2 trial
        self.linear_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_of_classes),
            nn.Sigmoid())

        # self.linear_layer = nn.Sequential(
        #     nn.Dropout(0.2),
        #     nn.Linear(conv_output_size, num_of_classes),
        #     nn.Sigmoid())

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # weight
        weights = torch.tensor([4.0, 1.0])

        self.loss_fn = nn.CrossEntropyLoss(weight=weights)

        self.f1_value = F1Score(task="multiclass", num_classes=num_of_classes, average='macro')
        self.precision_value = Precision(task="multiclass", num_classes=num_of_classes,
                                         average='macro')
        self.recall_value = Recall(task="multiclass", num_classes=num_of_classes, average='macro')
        self.accuracy_value = Accuracy(task="multiclass", num_classes=num_of_classes)

    def training_step(self, batch, batch_idx):
        x, labels = batch
        output = self(x)
        loss = self.loss_fn(output, labels)

        return loss

    def _get_conv_output_size(self, input_size):
        # 임의의 입력 텐서에 대해 Conv layer의 출력 크기 계산
        x = torch.randn(1, 3, *input_size)
        x = self.conv_layer(x)
        output_size = x.size(1) * x.size(2) * x.size(3)
        return output_size

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        output = self.linear_layer(x)

        # output = self.linear_layer(x_2)

        return output

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        prediction = torch.argmax(y_hat, dim=1)

        loss = self.loss_fn(y_hat, y)
        f1 = self.f1_value(prediction, y)
        precision = self.precision_value(prediction, y)
        accuracy = self.accuracy_value(prediction, y)
        recall = self.recall_value(prediction, y)

        # save data for tensorboard
        self.log("loss", loss)
        self.log("f1_value", f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("precision_value", precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log("accuracy_value", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("recall_value", recall, prog_bar=True, on_step=False, on_epoch=True)
        # return matrix value for further analysis
        return {'test_loss': loss, 'test_f1': f1, 'test_precision': precision, 'test_accuracy': accuracy,
                'test_recall': recall}

