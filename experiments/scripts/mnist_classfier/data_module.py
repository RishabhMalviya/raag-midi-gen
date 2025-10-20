# pylint: disable=arguments-differ
# pylint: disable=unused-argument
# pylint: disable=abstract-method
import lightning.pytorch as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from raag_midi_gen.paths import RAW_DATA_DIR


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=3):
        """
        Initialization of inherited lightning data module
        """
        super().__init__()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.batch_size = batch_size
        self.num_workers = num_workers

        # transforms for images
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )


    def _create_data_loader(self, df):
        """
        Generic data loader function

        :param df: Input tensor

        :return: Returns the constructed dataloader
        """
        return DataLoader(df, batch_size=self.batch_size, num_workers=self.num_workers)
    

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        :param stage: Stage - training or testing
        """
        train_data = datasets.MNIST(
            RAW_DATA_DIR, download=True, train=True, transform=self.transform
        )
        test_data = datasets.MNIST(
            RAW_DATA_DIR, download=True, train=False, transform=self.transform
        )

        self.df_test = test_data
        self.df_train, self.df_val = random_split(train_data, [55000, 5000])


    def train_dataloader(self):
        """
        :return: output - Train data loader for the given input
        """
        return self._create_data_loader(self.df_train)


    def val_dataloader(self):
        """
        :return: output - Validation data loader for the given input
        """
        return self._create_data_loader(self.df_val)


    def test_dataloader(self):
        """
        :return: output - Test data loader for the given input
        """
        return self._create_data_loader(self.df_test)