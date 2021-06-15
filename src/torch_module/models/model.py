"""
- you can use lightning module as nn.module


Threads
- study nn.module 

"""
import pytorch_lightning as pl
from torch import nn

class ClassificationTask(pl.LightningModule): 
    def __init__(self, backbone, num_classes):
        super().__init__()

        # lightning modules are best structured as systems
        self.backbone = backbone 
        self.linear = nn.Sequential(
            nn.Linear(backbone.latent_dim, 64), 
            nn.ReLU(),
            nn.Linear(64, num_classes), 
        )
        self.save_hyperparameters()

    def forward(self, x):
        """ 
        forward defines the prediction / inference actions 
        use for inference only - keep it separate from training step 
        """
        embedding = self.backbone(x)
        output = self.linear(embedding)
        return output

    def training_step(self, batch, batch_idx):
        """
        - defines the train loop
        - should return loss
        - independent of forward
        """
        x, y = batch 
        self.print('x shape', x.shape)
        self.print('y shape', y.shape)
        loss, acc = self.shared_step(batch)
        """
        logs metrics for each training step and average across each epoch to progress bar and logger (imagine tensorboard)
        logs to tensorboard by default 
        """
        self.log('train_acc', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        define optimizers and lr scheculers
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {
            'optimizer': optimizer, 
            'lr_scheduler': {
                'scheduler': ReduceLROnPlateau(optimizer),
                'monitor': 'metric_to_track',
            }
        }


    def validation_step(self, batch, batch_idx): 
        """
        override this method and add a validation loop
        """
        x, y = batch 
        metrics = { 'acc': acc, 'val_loss': loss }
        self.log_dict(metrics)
        loss = self.shared_step(batch)
        self.log('val_loss', loss)
        self.log('val_acc', acc)


    def test_step(self, batch, batch_idx): 
        """
        similar to train and validaiton except it's only called when .test()
        fun fact: 
        if you call trainer.test() after trainer.fit, it'll call the best weights
        """

    def _shared_eval_step(self, batch, batch_idx):
        """
        shared step 
        """
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        return loss, acc


backbone = 'epic'

# If you seprate model from task, you don't need to have model as LightningModule in inference.
# It can be onnx, jit or anything else
task = ClassificationTask(backbone)
trainer.fit(task, dl)
task.model_size

from pytorch_lightning.metrics import functional as FM
    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        acc = FM.accuracy(y_hat, y)
        return loss, acc



)

"""

For example, we couple the optimizer with a model because the majority of models require a specific optimizer with a specific learning rate scheduler to work well.
Instead, be explicit in your init
make sure to tune the number of workers for maximum efficiency.
"""

def __init__(self, lr, batch_size)

