import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch import nn
from Transformers_for_peptide_Classification.Core import (
    TPrep,
    Encoder,
    MAE_MaskingRaw,
    MAE_Decoder,
    MAE_CreateDecoderInput_Raw,
    MAE_CalcLoss_Raw,
    CrossAttention,
)


class MultiModal_TimeSeriesDescriptor_Pretraining(pl.LightningModule):
    def __init__(self, hp, hp_lr, hp_mask):
        super().__init__()
        self.save_hyperparameters()
        self.lr = hp["learning_rate"]
        self.num_epochs = hp["num_epochs"]
        self.steps_per_epoch = hp["steps_per_epoch"]
        self.pct_start = hp_lr["pct_start"]
        self.div_factor = hp_lr["div_factor"]
        self.seq_length = hp["seq_length"]
        self.final_div_factor = hp_lr["final_div_factor"]
        self.k0 = hp["patch_size"]
        self.batch_size = hp["batch_size"]
        self.input_size = hp[
            "input_size"
        ]  # length of original input sequence before patching
        self.k = hp["dimension"]
        self.alpha = hp["alpha"]

        self.num_disp = 5
        self.dips_output = torch.zeros(self.num_disp, self.seq_length, self.k0)
        self.disp_mask_id = torch.zeros(self.num_disp, self.seq_length)
        self.disp_orig_image = torch.zeros(self.num_disp, self.seq_length, self.k0)
        self.l = torch.zeros(self.num_disp)
        self.plot_freq = 20

        self.encoder = Encoder(
            depth=hp["depth"],
            dimension=self.k,
            heads=hp["heads"],
            hidden_dim=hp["hidden_dim"],
            dropout=hp["dropout"],
        )
        self.PrepPatches = TPrep(
            input_dimension=self.k0,
            dimension=self.k,
            sequence_length=self.seq_length,
            Pretrained=hp["Pretrained"],
        )
        self.masking = MAE_MaskingRaw(
            dimension=self.k,
            seq_length=self.seq_length,
            mask_ratio=hp_mask["mask_ratio"],
            patch_size=self.k0,
        )
        self.cross_att = CrossAttention(
            self.k, self.k, hp["heads"], hp["dropout"], add_in_both_latent_space=False
        )
        self.create_decoder_input = MAE_CreateDecoderInput_Raw(
            dimension=self.k,
            sequence_length=self.seq_length,
        )
        self.decode = MAE_Decoder(
            depth=hp["depth_decoder"],
            input_dimension=self.k0,
            dimension=self.k,
            heads=hp["heads"],
            hidden_dim=hp["hidden_dim"],
            dropout=hp["dropout"],
        )
        mse_loss = nn.MSELoss()
        self.calc_loss = MAE_CalcLoss_Raw(
            alpha=hp["alpha"], batch_size=hp["batch_size"], loss_func=mse_loss
        )

        catch22_dropout = hp["dropout"]
        ## feed forward for catch25
        self.features_Dropout = nn.Dropout(catch22_dropout)
        self.layernorm_features = nn.LayerNorm(256)
        self.GELU = nn.GELU()
        self.fc1 = nn.Linear(hp["feature_dim"], 128)
        self.fc2 = nn.Linear(128, 256)

    def forward(self, x, features, l):
        b, t, k0 = x.size()
        orig_series = x.detach().clone()  # for plots
        CLS, x, pe = self.PrepPatches(
            x
        )  # retuns: CLS (already expanded), token_embedding (pe alreadz added), pe
        unmasked_embeddings, mask_list = self.masking(x, pe, l)
        # prepare features
        features = self.GELU(self.fc1(features))
        features = self.GELU(self.fc2(features))
        features = self.layernorm_features(features)
        features = self.features_Dropout(features)
        # cross attention
        CLS = self.cross_att(CLS, features[:, None, :])
        # encoding and decoding
        encoder_input = torch.cat(
            (CLS, unmasked_embeddings), dim=1
        )  # add CLS token for encoder input
        encoder_output = self.encoder(encoder_input)[
            :, 1:, :
        ]  # remove CLS token from encoder output
        decoder_inputs = self.create_decoder_input(encoder_output, mask_list)
        reconstructed = self.decode(decoder_inputs)
        return reconstructed, mask_list[2], mask_list[3], orig_series

    def training_step(self, batch, batch_idx):
        raw, catch = batch
        x, y, l = raw
        catch, _ = catch
        outputs, mask_id, unmask_id, orig_image = self.forward(
            x, catch, l
        )  # shapes: outputs: (b, t, k0), mask_id: (b, mask_seq_length), mask_type: (b), orig_image: (b, width, height)
        loss = self.calc_loss(outputs, mask_id, unmask_id, orig_image)
        self.log("train_loss", loss, sync_dist=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        print("validation step")
        raw, catch = batch
        x, _, l = raw
        catch, _ = catch
        outputs, mask_id, unmask_id, orig_image = self.forward(
            x, catch, l
        )  # shapes: outputs: (b, t, k0), mask_id: (b, mask_seq_length), mask_type: (b), orig_image: (b, width, height)
        loss = self.calc_loss(outputs, mask_id, unmask_id, orig_image)
        self.log("val_loss", loss, sync_dist=True)
        # extract last 5 images of val batch for every ten epochs to save
        if self.current_epoch % self.plot_freq == 0:
            self.disp_output = outputs[: self.num_disp, ...]
            self.disp_mask_id = mask_id[: self.num_disp]
            self.disp_orig_image = orig_image[: self.num_disp, ...]
            self.l = l[: self.num_disp]
        return loss

    def on_validation_epoch_end(self):
        if self.current_epoch % self.plot_freq == 0:
            x = []
            for i in range(self.num_disp):
                x.append(torch.arange(self.l[i]))
            fig, ax = plt.subplots(10, 1, figsize=(5, 15))
            for i in range(self.num_disp):
                series1 = (
                    self.disp_orig_image[i, :]
                    .reshape(self.input_size, 1)
                    .detach()
                    .cpu()
                )
                series3 = (
                    self.disp_output[i, :].reshape(self.input_size, 1).detach().cpu()
                )
                ax[2 * i].plot(x[i], series1[: self.l[i]])
                for k in range(self.disp_mask_id[i].size(dim=0)):
                    start = self.disp_mask_id[i][k].cpu()
                    ax[2 * i].plot(
                        x[i][start * self.k0 : start * self.k0 + self.k0],
                        series1[start * self.k0 : start * self.k0 + self.k0],
                        color="red",
                    )
                ax[2 * i + 1].plot(series3[: self.l[i]])
            self.logger.experiment.add_figure(
                "Validation_Plots", fig, self.current_epoch
            )
            plt.close(fig)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
