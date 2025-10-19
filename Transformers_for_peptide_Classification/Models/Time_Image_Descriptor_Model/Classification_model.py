import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import torch
import torchmetrics
from torch import nn
from Transformers_for_peptide_Classification.Core import (
    TPrep,
    CrossViT_Encoder,
    ClassifierCLS,
    CrossAttention,
)


class MultiModal_TimeSeriesImageDescriptor_Classifier(pl.LightningModule):
    def __init__(
        self,
        hp,
        hp_lr,
        model_name,
        encoder=None,
        PrepPatches_raw=None,
        PrepPatches_image=None,
        fc1=None,
        fc2=None,
        cross_att_raw=None,
        cross_att_img=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        num_classes = hp["num_classes"]
        self.lr = hp["learning_rate"]
        self.num_epochs = hp["num_epochs"]
        self.steps_per_epoch = hp["steps_per_epoch"]
        self.num_branches = hp["channels"]
        self.k0_raw = hp["input_dimension_list"][0]
        self.k0_image = hp["input_dimension_list"][1]
        self.feature_input_dim = hp["feature_input_dim"]
        self.model_name = model_name

        # learning rate settings
        self.pct_start = hp_lr["pct_start"]
        self.div_factor = hp_lr["div_factor"]
        self.final_div_factor = hp_lr["final_div_factor"]

        # use pretrained modules if provided
        if PrepPatches_raw is None:
            self.PrepPatches_raw = TPrep(
                input_dimension=self.k0_raw,
                dimension=hp["dimension_list"][0],
                sequence_length=hp["seq_length_raw"],
                Pretrained=hp["Pretrained"],
            )
        else:
            self.PrepPatches_raw = PrepPatches_raw

        if PrepPatches_image is None:
            self.PrepPatches_image = TPrep(
                input_dimension=self.k0_image,
                dimension=hp["dimension_list"][1],
                sequence_length=hp["seq_length_image"],
                Pretrained=hp["Pretrained"],
            )
        else:
            self.PrepPatches_image = PrepPatches_image

        if encoder is None:
            self.encoder = CrossViT_Encoder(hp)
        else:
            self.encoder = encoder
        if cross_att_raw is None:
            self.cross_att_raw = CrossAttention(
                256,
                hp["dimension_list"][0],
                4,
                hp["dropout"],
                add_in_both_latent_space=False,
            )
        else:
            self.cross_att_raw = cross_att_raw
        if cross_att_img is None:
            self.cross_att_img = CrossAttention(
                256,
                hp["dimension_list"][1],
                4,
                hp["dropout"],
                add_in_both_latent_space=False,
            )
        else:
            self.cross_att_img = cross_att_img
        # create classifier
        self.Classifier = nn.ModuleList(
            [
                ClassifierCLS(num_classes, hp["dimension_list"][0]),
                ClassifierCLS(num_classes, hp["dimension_list"][1]),
            ]
        )
        # create loss function
        self.loss_cross_entropy = nn.CrossEntropyLoss(
            label_smoothing=hp["label_smoothing"]
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

        self.pre_conf_val = []  # list to store predictions for confusion matrix
        self.lab_conf_val = []  # list to store labels for confusion matrix
        self.pre_conf_test = []  # list to store predictions for confusion matrix
        self.lab_conf_test = []  # list to store labels for confusion matrix

        self.cm = torchmetrics.ConfusionMatrix(
            task="multiclass",
            num_classes=num_classes,
        )
        catch22_dropout = hp["dropout"]
        ## feed forward for catch25
        self.catch22Dropout = nn.Dropout(catch22_dropout)
        self.layernormCatch22 = nn.LayerNorm(256)
        self.GELU = nn.GELU()
        if fc1 is None:
            self.fc1 = nn.Linear(self.feature_input_dim, 128)
        else:
            self.fc1 = fc1
        if fc2 is None:
            self.fc2 = nn.Linear(128, 256)
        else:
            self.fc2 = fc2
        self.fc3 = nn.Linear(256, num_classes)
        
        self.display_frequency = 20

    def forward(self, x, feature_vector):
        # create embeddings and CLS tokens
        CLS_raw, embedded_raw, _ = self.PrepPatches_raw(x[0])
        CLS_image, embedded_image, _ = self.PrepPatches_image(x[1])
        # prepare feature_vector
        feature_vector = self.GELU(self.fc1(feature_vector))
        feature_vector = self.GELU(self.fc2(feature_vector))
        feature_vector = self.layernormCatch22(feature_vector)
        feature_vector = self.catch22Dropout(feature_vector)
        # cross attention
        CLS_raw = self.cross_att_raw(CLS_raw, feature_vector[:, None, :])
        CLS_image = self.cross_att_img(CLS_image, feature_vector[:, None, :])
        # create branch list
        branch_list = [
            torch.cat((CLS_raw, embedded_raw), dim=1),
            torch.cat((CLS_image, embedded_image), dim=1),
        ]
        # encode the sequences
        encoded = self.encoder(branch_list)
        # classify
        output = [
            self.Classifier[i](encoded[i][:, 0, :]) for i in range(self.num_branches)
        ]
        # average over classifications
        output = output[0] + output[1] + self.fc3(feature_vector)
        return output

    def training_step(self, batch, batch_idx):
        raw, image, feature = batch
        b, t0, k0 = raw[0].shape
        x = [raw[0], image[0]]
        y = raw[1]
        output = self.forward(x, feature)
        loss = self.loss_cross_entropy(output, y)
        # log metric
        self.log("train_loss", loss, sync_dist=True, on_step=True)
        self.train_accuracy(output, y)
        self.log("train_acc_macro", self.train_accuracy, sync_dist=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        raw, image, feature = batch
        b, t0, k0 = raw[0].shape
        x = [raw[0], image[0]]
        y = raw[1]
        output = self.forward(x, feature)
        loss = self.loss_cross_entropy(output, y)
        self.log("val_loss", loss, sync_dist=True)
        self.val_accuracy(output, y)
        self.log("val_acc_macro", self.val_accuracy, sync_dist=True)
        _, predicted = torch.max(output.data, 1)
        self.pre_conf_val.append(predicted)
        self.lab_conf_val.append(y)
        return loss

    def on_validation_epoch_end(self):
        if self.current_epoch % self.display_frequency == 0:
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
        raw, image, feature = batch
        b, t0, k0 = raw[0].shape
        x = [raw[0], image[0]]
        y = raw[1]
        output = self.forward(x, feature)
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
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # 'step' for step-based scheduling
                "frequency": 1,  # how often to apply
            },
        }
