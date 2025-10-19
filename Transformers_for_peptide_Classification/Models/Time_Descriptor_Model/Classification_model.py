import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
from torch import nn
from Transformers_for_peptide_Classification.Core import (
    TPrep,
    Encoder,
    ClassifierCLS,
    CrossAttention,
)

class MultiModal_TimeSeriesDescriptor_Classifier(pl.LightningModule):
    def __init__(
        self,
        hp,
        hp_lr,
        model_name,
        encoder=None,
        PrepPatches=None,
        cross_att=None,
        fc1=None,
        fc2=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        num_classes = hp["num_classes"]
        self.lr = hp["learning_rate"]
        self.num_epochs = hp["num_epochs"]
        self.steps_per_epoch = hp["steps_per_epoch"]
        self.pct_start = hp_lr["pct_start"]
        self.div_factor = hp_lr["div_factor"]
        self.final_div_factor = hp_lr["final_div_factor"]
        self.model_name = model_name

        self.pre_conf_test = []
        self.lab_conf_test = []
        self.pre_conf_val = []
        self.lab_conf_val = []

        if encoder is None:
            self.encoder = Encoder(
                depth=hp["depth"],
                dimension=hp["dimension"],
                heads=hp["heads"],
                hidden_dim=hp["hidden_dim"],
                dropout=hp["dropout"],
            )
        else:
            self.encoder = encoder
        if PrepPatches is None:
            self.PrepPatches = TPrep(
                input_dimension=hp["input_dimension"],
                dimension=hp["dimension"],
                sequence_length=hp["seq_length"],
                Pretrained=hp["Pretrained"],
            )
        else:
            self.PrepPatches = PrepPatches
        if cross_att is None:
            self.cross_att = CrossAttention(
                hp["dimension"],
                hp["dimension"],
                hp["heads"],
                hp["dropout"],
                add_in_both_latent_space=False,
            )
        else:
            self.cross_att = cross_att
        self.ClassifierCLS = ClassifierCLS(
            num_classes=num_classes,
            dimension=hp["dimension"],
        )

        self.train_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.val_accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_accuracy_macro = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, average="macro"
        )
        self.test_accuracy_micro = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes, average="micro"
        )

        self.cm = torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
        )
        catch22_dropout = hp["dropout"]
        ## feed forward for catch25
        self.feautures_Dropout = nn.Dropout(catch22_dropout)
        self.layernorm_features = nn.LayerNorm(256)
        self.GELU = nn.GELU()
        self.fc1 = nn.Linear(hp['feature_dim'], 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.loss_cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=hp["label_smoothing"]
        )

    def forward(self, x, features):
        CLS, x, _ = self.PrepPatches(
            x
        )  # retuns: CLS (already expanded), token_embedding (pe alreadz added), pe
        # prepare feature
        features = self.GELU(self.fc1(features))
        features = self.GELU(self.fc2(features))
        features = self.layernorm_features(features)
        features = self.feautures_Dropout(features)
        # cross attention
        CLS = self.cross_att(CLS, features[:, None, :])
        # encoding and decoding
        encoder_input = torch.cat((CLS, x), dim=1)
        encoded = self.encoder(encoder_input)
        classified = self.ClassifierCLS(encoded[:, 0, :])  # classification on CLS token
        classified_catch = self.fc3(features)
        return classified + classified_catch

    def training_step(self, batch, batch_idx):
        raw, catch = batch
        x, y, l = raw
        catch, _ = catch
        output = self.forward(x, catch)
        loss = self.loss_cross_entropy(output, y)
        # log metric
        self.log("train_loss", loss, sync_dist=True, on_step=True)
        self.train_accuracy(output, y)
        self.log("train_acc_macro", self.train_accuracy, sync_dist=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        raw, catch = batch
        x, y, l = raw
        catch, _ = catch
        output = self.forward(x, catch)
        loss = self.loss_cross_entropy(output, y)
        self.log("val_loss", loss, sync_dist=True)
        self.val_accuracy(output, y)
        self.log("val_acc_macro", self.val_accuracy, sync_dist=True)
        _, predicted = torch.max(output.data, 1)
        self.pre_conf_val.append(predicted)
        self.lab_conf_val.append(y)
        return loss

    def on_validation_epoch_end(self):
        if self.current_epoch % 20 == 0:
            outputs = torch.cat(self.pre_conf_val, dim=0)
            targets = torch.cat(self.lab_conf_val, dim=0)
            cf_matrix = self.cm(outputs, targets)
            # Normalize confusion matrix
            cf_matrix = (cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis]).cpu()

            fig, ax = plt.subplots(figsize=(10, 7))
            # Create a heatmap without numbers
            sns.heatmap(cf_matrix, annot=False, ax=ax, cmap="Blues")

            # Label the axes
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            # Log confusion matrix
            self.logger.experiment.add_figure(
                "Validation Confusion Matrix", fig, self.current_epoch
            )
            plt.close(fig)
        self.pre_conf_val = []
        self.lab_conf_val = []

    def test_step(self, batch, batch_idx):
        raw, catch = batch
        x, y, l = raw
        catch, _ = catch
        output = self.forward(x, catch)
        loss = self.loss_cross_entropy(output, y)
        self.log("test_loss", loss)
        # log step metric
        _, predicted = torch.max(output.data, 1)
        self.pre_conf_test.append(predicted)
        self.lab_conf_test.append(y)
        self.test_accuracy_macro(output, y)
        self.log("test_acc_macro", self.test_accuracy_macro, sync_dist=True)
        self.test_accuracy_micro(output, y)
        self.log("test_acc_micro", self.test_accuracy_micro)
        return {"y_true": output, "y_pred": y}

    def on_test_epoch_end(self):
        outputs = torch.cat(self.pre_conf_test, dim=0)
        targets = torch.cat(self.lab_conf_test, dim=0)
        cf_matrix = self.cm(outputs, targets)
        torch.save(cf_matrix, "logs/cm_" + self.model_name + ".pt")
        # Normalize confusion matrix
        cf_matrix = (cf_matrix / cf_matrix.sum(axis=1)[:, np.newaxis]).cpu()

        fig, ax = plt.subplots(figsize=(10, 7))
        # Create a heatmap without numbers
        sns.heatmap(cf_matrix, annot=False, ax=ax, cmap="Blues")

        # Label the axes
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Log confusion matrix
        self.logger.experiment.add_figure(
            "Test Confusion Matrix", fig, self.current_epoch
        )
        plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Use the custom scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lr,
            steps_per_epoch=int(self.steps_per_epoch),
            epochs=int(self.num_epochs),
            pct_start=self.pct_start,
        )
        # scheduler = LinearWarmupScheduler(optimizer, max_lr=self.lr, epochs=int(self.num_epochs), steps_per_epoch=int(self.steps_per_epoch), pct_start=self.pct_start, div_factor=self.div_factor, final_div_factor=self.final_div_factor)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 'step' for step-based scheduling
                "frequency": 1,  # how often to apply
            },
        }
