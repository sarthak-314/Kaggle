import torch 
from src.data.datasets import CompDataset

class CompDataModule(pl.LightningDataModule):
    """
    CompDataModule for CompDataset. 

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """
    def __init__(self, processed_dataframes, data_transforms, batch_size=32): 
        super().__init__()
        self.dataframes = processed_dataframes
        self.data_transforms = data_transforms 
        self.batch_size = batch_size 

    def prepare_data(self, stage): 
        """
        things to do on 1 gpu/tpu not
        download / tokenize data
        :param stage: like fit, test
        """
        pass

    def setup(self, stage): 
        """
        was for split and stuff but pass
        """
        pass
    
    def train_dataloader(self):
        # read train dataframe
        train = self.processed['train']
        # build torch dataset
        train_ds = GenericDataset(df=train, trainsforms=self.data_transforms['train'], df_type='train')
        # build train loader
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=self.batch_size, 
            num_workers = self.num_workers, # you should optimize this 
            pin_memory = torch.cuda.is_available(), 
            shuffle = True, drop_last = True, 
        )
        return train_loader

    def val_dataloader(self):
        valid = self.processed['valid']
        valid_ds = GenericDataset(df=valid, transforms=self.data_transforms['valid'], df_type='valid')
        val_loader = torch.utils.data.DataLoader(
            valid_ds, batch_size=self.batch_size, 
            num_workers = self.num_workers, 
            pin_memory = torch.cuda.is_available(), 
            shuffle = True, drop_last = False, 
        )
        return val_loader

    def test_dataloader(self): 
        test = self.processed['test']
        test_ds = GenericDataset(df=test, transforms=self.data_transforms['test'], df_type='test')
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=torch.cuda.is_available(), 
            shuffle=False, drop_last=False,
        )
        return test_loader

    def teardown(self, stage):
        """
        used to cleanup when the run is finished
        """
        pass


"""
⚒ Ideas & Improvements ⚒
-------------------------

- add dataframes metadata like num_classes as @property 
- 
"""



