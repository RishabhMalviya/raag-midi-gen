# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method
import torch
import lightning.pytorch as pl
from torch.nn import functional as F
from torchmetrics.functional import accuracy


class MNISTClassifier(pl.LightningModule):
    def _cross_entropy_loss(self, logits, labels):
        """
        Initializes the loss function

        :return: output - Initialized cross entropy loss function
        """
        return F.nll_loss(logits, labels)

    def __init__(self, learning_rate=0.01):
        """
        Initializes the network
        """
        super().__init__()

        self.save_hyperparameters()

        # mnist images are (1, 28, 28) (channels, width, height)
        self.optimizer = None
        self.scheduler = None
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)
        self.learning_rate = learning_rate
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, x):
        """
        :param x: Input data

        :return: output - mnist digit label for the input image
        """
        batch_size = x.size()[0]

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)

        # layer 1 (b, 1*28*28) -> (b, 128)
        x = self.layer_1(x)
        x = torch.relu(x)

        # layer 2 (b, 128) -> (b, 256)
        x = self.layer_2(x)
        x = torch.relu(x)

        # layer 3 (b, 256) -> (b, 10)
        x = self.layer_3(x)

        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        :param train_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Training loss
        """
        x, y = train_batch
        logits = self.forward(x)
        loss = self._cross_entropy_loss(logits, y)
        self.log("Training Loss", loss, prog_bar=True, enable_graph=True)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches

        :param val_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - valid step loss
        """
        x, y = val_batch
        logits = self.forward(x)
        loss = self._cross_entropy_loss(logits, y)
        self.val_outputs.append(loss)
        return {"val_step_loss": loss}

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the accuracy of the model

        :param test_batch: Batch data
        :param batch_idx: Batch indices

        :return: output - Testing accuracy
        """
        x, y = test_batch
        output = self.forward(x)
        _, y_hat = torch.max(output, dim=1)
        test_acc = accuracy(y_hat.cpu(), y.cpu(), task="multiclass", num_classes=10)
        self.test_outputs.append(test_acc)
        return {"test_acc": test_acc}

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        :return: output - Initialized optimizer and scheduler
        """
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]

    def on_validation_epoch_end(self):
        """
        Computes average validation loss
        """
        avg_loss = torch.stack(self.val_outputs).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.val_outputs.clear()

    def on_test_epoch_end(self):
        """
        Computes average test accuracy score
        """
        avg_test_acc = torch.stack(self.test_outputs).mean()
        self.log("avg_test_acc", avg_test_acc, sync_dist=True)
        self.test_outputs.clear()
