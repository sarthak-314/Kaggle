from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import pytorch_lightning as pl
from torch import nn
import torchmetrics

class ClassificationTask(pl.LightningModule): 
    def __init__(self, backbone, final_layer):
        super().__init__()

        # lightning modules are best structured as systems
        self.backbone = backbone 
        self.linear = final_layer        
        self.save_hyperparameters()
        
        # torchmetrics stuff
        self.train_acc = torchmetrics.Accuracy()
        self.valid_acc = torchmetrics.Accuracy()

    def forward(self, x):
        """ 
        forward defines the prediction / inference actions 
        use for inference only - keep it separate from training step 
        """
        embedding = self.backbone(x)
        output = self.linear(embedding)
        return output

    def training_step(self, batch, _):
        """
        - defines the train loop
        - should return loss
        - independent of forward
        """
        y = batch.pop('target')
        y_hat = self.forward(**batch) 
        loss = F.cross_entropy(y_hat, y)
        acc = self.train_acc(y, y_hat)
        self.log_dict({'train/loss': loss, 'train/acc': acc})
        return loss
    
    def configure_optimizers(self):
        """
        define optimizers and lr scheculers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': CosineAnnealingWarmRestarts(optimizer, T_0=5),
                'monitor': 'val/acc',
            }
        }

    def validation_step(self, batch, _): 
        y = batch.pop('target')
        y_hat = self.forward(**batch) 
        loss = F.cross_entropy(y_hat, y)
        acc = self.train_acc(y, y_hat)
        self.log_dict({'train/loss': loss, 'train/acc': acc})
        return loss

    def test_step(self, batch, _): 
        y_pred = self.forward(**batch) 
        return y_pred


