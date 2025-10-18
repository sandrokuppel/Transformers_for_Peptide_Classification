import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch import nn
from Transformers_for_peptide_Classification import (
    TPrep, 
    CrossViT_Encoder,
    MAE_MaskingImage, 
    MAE_MaskingRaw,
    recreate_single_picture,
    MAE_CreateDecoderInput_Wavelets,
    MAE_CreateDecoderInput_Raw,
    CrossViT_Decoder,
    MAE_CalcLoss_Wavelets,
    MAE_CalcLoss_Raw,
    CrossAttention
)


class MAE_MultiModal_WaveletsRaw_Pretraining(pl.LightningModule):
    def __init__(self, hp, hp_lr, hp_mask):
        super().__init__()
        self.save_hyperparameters()

        self.k = hp["dimension_list"]
        self.height = hp["height"]
        self.width = hp["width"]
        self.lr = hp["learning_rate"]
        self.num_epochs = hp["num_epochs"]
        self.steps_per_epoch = hp["steps_per_epoch"]
        self.pct_start = hp_lr['pct_start']
        self.div_factor = hp_lr['div_factor']
        self.final_div_factor = hp_lr['final_div_factor']
        self.num_branches = hp["channels"]
        self.k0_raw = hp["input_dimension_list"][0]
        self.k0_image = hp["input_dimension_list"][1]
        self.seq_length_image = hp["seq_length_image"]
        self.seq_length_raw = hp["seq_length_raw"]
        self.max_length_raw = hp["max_length_raw"]
        self.patch_size_image = hp["patch_size_image"]
        self.feature_input_dim = hp["feature_input_dim"]

        self.PrepPatches_raw = TPrep(input_dimension = self.k0_raw, 
                                     dimension = hp['dimension_list'][0],
                                     sequence_length = hp['seq_length_raw'],
                                     Pretrained=hp['Pretrained'])
        self.PrepPatches_image = TPrep(input_dimension = self.k0_image, 
                                     dimension = hp['dimension_list'][1],
                                     sequence_length = hp['seq_length_image'],
                                     Pretrained=hp['Pretrained'])
        self.encoder = CrossViT_Encoder(hp)
        self.mask_raw = MAE_MaskingRaw(
            dimension=hp['dimension_list'][0],
            seq_length=hp['seq_length_raw'],
            mask_ratio=hp_mask['mask_ratio_raw'],
            patch_size=hp['patch_size_raw']
        )
        self.mask_image = MAE_MaskingImage(
            dimension=hp['dimension_list'][1],
            seq_length=hp['seq_length_image'],
            mask_ratio=hp_mask['mask_ratio_image']
        )

        self.create_decoder_input = nn.ModuleList([
            MAE_CreateDecoderInput_Raw(
                dimension=hp['dimension_list'][0], 
                sequence_length=self.seq_length_raw
                ),
            MAE_CreateDecoderInput_Wavelets(
                dimension=hp['dimension_list'][1], 
                sequence_length=self.seq_length_image
                )
        ])
        self.decoder = CrossViT_Decoder(hp)
        self.mse_loss= nn.MSELoss()
        self.calc_loss = nn.ModuleList([
            MAE_CalcLoss_Raw(
                alpha = hp['alpha'],
                batch_size = hp['batch_size'],
                loss_func = self.mse_loss
            ),
            MAE_CalcLoss_Wavelets(
                alpha = hp['alpha'],
                batch_size = hp['batch_size'],
                loss_func = self.mse_loss
            )
        ])

        self.num_disp = 5	
        self.display_frequency = 20

        self.disp_output_pic = torch.zeros(self.num_disp, self.seq_length_image, self.k0_image)
        self.disp_mask_id_pic = torch.zeros(self.num_disp, self.seq_length_image)
        self.disp_orig_pic = torch.zeros(self.num_disp, self.seq_length_image, self.k0_image)

        self.disp_output_raw = torch.zeros(self.num_disp, self.seq_length_raw, self.k0_raw)
        self.disp_mask_id_raw = torch.zeros(self.num_disp, self.seq_length_raw)
        self.disp_orig_raw = torch.zeros(self.num_disp, self.seq_length_raw, self.k0_raw)
        self.l = torch.zeros(self.num_disp)

        catch22_dropout = hp["dropout"]
        ## feed forward for catch25
        self.catch22Dropout = nn.Dropout(catch22_dropout)
        self.layernormCatch22 = nn.LayerNorm(256)
        self.GELU = nn.GELU()

        self.fc1 = nn.Linear(self.feature_input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.cross_att_raw = CrossAttention(self.k[0], 256, hp["heads_list"][0], hp["dropout"], add_in_both_latent_space=False)
        self.cross_att_wavelet = CrossAttention(self.k[1], 256, hp["heads_list"][1], hp["dropout"], add_in_both_latent_space=False)

    def forward(self, x, feature_vector, l):
        # save original input for loss calculation
        original = [x[i].detach().clone() for i in range(self.num_branches)]               
        # create embeddings and CLS tokens
        CLS_raw, embedded_raw, pe_raw = self.PrepPatches_raw(x[0])
        CLS_image, embedded_image, pe_image = self.PrepPatches_image(x[1])
        # prepare feature vector
        feature_vector = self.GELU(self.fc1(feature_vector))
        feature_vector = self.GELU(self.fc2(feature_vector))
        feature_vector = self.layernormCatch22(feature_vector)
        feature_vector = self.catch22Dropout(feature_vector)
        # cross attention 
        CLS_raw = self.cross_att_raw(CLS_raw, feature_vector[:,None,:])
        CLS_image = self.cross_att_wavelet(CLS_image, feature_vector[:,None,:])
        # mask raw time series and pictures
        unmasked_embeddings_raw, mask_list_raw = self.mask_raw(embedded_raw, pe_raw, l)
        unmasked_embeddings_image, mask_list_image = self.mask_image(embedded_image, pe_image)
        mask_list = [mask_list_raw, mask_list_image]
        # create branch list
        branch_list = [torch.cat((CLS_raw, unmasked_embeddings_raw), dim=1), torch.cat((CLS_image, unmasked_embeddings_image), dim=1)]
        # encode the sequences
        encoded = self.encoder(branch_list)
        # create decoder input
        decoder_input = [self.create_decoder_input[i](encoded[i][:,1:,:], mask_list[i]) for i in range(self.num_branches)]
        # recreate original input
        output = self.decoder(decoder_input)
        # create list containing [mask_index, unmask_index] for both branches -> for loss calc
        index_list = [mask_list[i][2:] for i in range(self.num_branches)]   
        return output, index_list, original

    def training_step(self, batch, batch_idx):
        raw, image, feature = batch                            
        b, t0, k0 = raw[0].shape
        x = [raw[0], image[0]]
        l = raw[2]
        outputs, index_list, original = self.forward(x, feature[0], l)
        # calculate individual losses for each branch
        loss_list = [self.calc_loss[i](outputs[i], index_list[i][0], index_list[i][1], original[i]) for i in range(self.num_branches)]
        self.log('train_loss_raw', loss_list[0], sync_dist=True)
        self.log('train_loss_img', loss_list[1], sync_dist=True)
        # create final loss by summing up the individual losses
        loss = loss_list[0] + loss_list[1]
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        raw, image, feature = batch                            
        b, t0, k0 = raw[0].shape
        x = [raw[0], image[0]]
        l = raw[2]
        outputs, index_list, original = self.forward(x, feature[0], l)
        # calculate individual losses for each branch
        loss_list = [self.calc_loss[i](outputs[i], index_list[i][0], index_list[i][1], original[i]) for i in range(self.num_branches)]
        self.log('val_loss_raw', loss_list[0], sync_dist=True)
        self.log('val_loss_img', loss_list[1], sync_dist=True)
        # create final loss by summing up the individual losses
        loss = loss_list[0] + loss_list[1]
        self.log('val_loss', loss, sync_dist=True)
        # extract last 5 images of val batch for every ten epochs to save 
        if self.current_epoch % self.display_frequency == 0:
            # raw data:
            self.disp_output_raw = outputs[0][:self.num_disp,...]
            self.disp_mask_id_raw = index_list[0][0][:self.num_disp] 
            self.disp_orig_raw = original[0][:self.num_disp,...] 
            self.l = l[:self.num_disp]
            # images:
            self.disp_output_pic = outputs[1][:self.num_disp,...]
            self.disp_mask_id_pic = index_list[1][0][:self.num_disp] 
            self.disp_orig_pic = original[1][:self.num_disp,...]
        return loss
    
    def on_validation_epoch_end(self):
        '''
        Create and log plots/pictures showing the reconstruction performance
        Plot some example time series and images with masked parts and the reconstruction
        '''
        if (self.current_epoch % self.display_frequency == 0):
            # raw data:
            x = []
            for i in range(self.num_disp):
                x.append(torch.arange(self.l[i]))
            fig, ax = plt.subplots(10, 1, figsize=(5, 15))
            for i in range(self.num_disp):
                series1 = self.disp_orig_raw[i,:].reshape(self.max_length_raw, 1).detach().cpu()
                series3 = self.disp_output_raw[i,:].reshape(self.max_length_raw, 1).detach().cpu()
                ax[2*i].plot(x[i],series1[:self.l[i]])
                for k in range(self.disp_mask_id_raw[i].size(dim=0)):
                    start = self.disp_mask_id_raw[i][k].cpu() - 1 # cause CLS token was removed
                    ax[2*i].plot(x[i][start*self.k0_raw:start*self.k0_raw+self.k0_raw], series1[start*self.k0_raw:start*self.k0_raw+self.k0_raw], color='red')
                ax[2*i+1].plot(series3[:self.l[i]])
            self.logger.experiment.add_figure('Val_Plots_Raw', fig, self.current_epoch)
            plt.close(fig)
            # pictures:
            mask_image = self.disp_orig_pic.detach().clone()
            for i in range(self.num_disp):
                mask_image[i, (self.disp_mask_id_pic[i]-1),:] = 1
            fig, ax = plt.subplots(self.num_disp, 3, figsize=(5, 15))
            for i in range(self.num_disp):
                im1 = recreate_single_picture(self.disp_orig_pic[i,:].detach(), self.height, self.width, self.patch_size_image)
                im2 = recreate_single_picture(mask_image[i,:].detach(), self.height, self.width, self.patch_size_image)
                im3 = recreate_single_picture(self.disp_output_pic[i,:].detach(), self.height, self.width, self.patch_size_image)
                ax[i, 0].imshow(im1)
                ax[i, 0].axis('off')
                ax[i, 1].imshow(im2)
                ax[i, 1].axis('off')
                ax[i, 2].imshow(im3)
                ax[i, 2].axis('off')
            self.logger.experiment.add_figure('Val_Pictures', fig, self.current_epoch)
            plt.close(fig)
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # Use the custom scheduler
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.lr, steps_per_epoch=int(self.steps_per_epoch), epochs=int(self.num_epochs), pct_start=self.pct_start)
        #scheduler = LinearWarmupScheduler(optimizer, max_lr=self.lr, epochs=int(self.num_epochs), steps_per_epoch=int(self.steps_per_epoch), pct_start=self.pct_start, div_factor=self.div_factor, final_div_factor=self.final_div_factor)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # 'step' for step-based scheduling
                'frequency': 1,  # how often to apply
            }
        }
 