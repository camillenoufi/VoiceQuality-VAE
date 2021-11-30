# Copyright (c) 2021 Rui Shu, modified by Camille Noufi 
import argparse
import numpy as np
import os
# import tensorflow as tf
import torch
from codebase import utils as ut
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image

def train(model, train_loader, validation_loader, device, tqdm, writer,
          iter_max=np.inf, iter_save=np.inf,
          model_name='model', y_status='none', reinitialize=False):
    # Optimization
    if reinitialize:
        model.apply(ut.reset_weights)
    min_lr = 1e-8
    epochs_max = 100
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience=5, verbose=True, min_lr=min_lr)
    
    total_params = sum(p.numel() for p in model.parameters()) # if p.requires_grad) #remove requires_grad if want all params
    print(f'Trainable Params: {total_params}')

    # init logs
    train_loss_arr = []
    valid_loss_arr = []
    rec_loss_arr = []
    kld_arr = []

    i = 0
    epoch = 0
    with tqdm(total=epochs_max) as pbar:
        while True:
            model.train()
            train_loss = 0
            valid_loss = 1e6
            rec_loss = 0
            kld = 0
            for batch_idx, (nu, xu, lu) in enumerate(train_loader):
                #lu[0] = gender, lu[1] = id, lu[2]=phrase, lu[3] = technique, lu[4] = vowel
                i += 1 # i is num of gradient steps taken by end of loop iteration
                optimizer.zero_grad()
                # import ipdb; ipdb.set_trace()
                if y_status == 'none':
                    loss, summaries = model.loss(xu.float())

                # elif y_status == 'semisup':
                #     xu = torch.bernoulli(xu.to(device).reshape(xu.size(0), -1))
                #     yu = yu.new(np.eye(10)[yu]).to(device).float()
                #     # xl and yl already preprocessed
                #     xl, yl = labeled_subset
                #     xl = torch.bernoulli(xl)
                #     loss, summaries = model.loss(xu, xl, yl)

                #     # Add training accuracy computation
                #     pred = model.cls(xu).argmax(1)
                #     true = yu.argmax(1)
                #     acc = (pred == true).float().mean()
                #     summaries['class/acc'] = acc

                # elif y_status == 'fullsup':
                #     # Janky code: fullsup is only for SVHN
                #     # xu is not bernoulli for SVHN
                #     xu = xu.to(device).reshape(xu.size(0), -1)
                #     yu = yu.new(np.eye(10)[yu]).to(device).float()
                #     loss, summaries = model.loss(xu, yu)

                loss.backward()
                optimizer.step()

                # Log summaries
                # if i % 50 == 0: ut.log_summaries(writer, summaries, i)

                # Save model
                if i % iter_save == 0:
                    ut.save_model_by_name(model, i, 
                                        train_loss_arr[:-1], valid_loss_arr,
                                        rec_loss_arr, kld_arr)

                # if i == iter_max:
                #     return
            
            #log average training loss for batch
            train_loss += loss/len(train_loader)
            train_loss_arr.append(train_loss.detach().numpy())

            rec_loss += summaries['gen/rec']/len(train_loader)
            rec_loss_arr.append(rec_loss.detach().numpy())

            kld += summaries['gen/kl_z']/len(train_loader)
            kld_arr.append(kld.detach().numpy())
                
            # forward pass over validation set at end of each epoch
            model.eval()
            valid_loss = 0
            with torch.no_grad():
                for batch_idx, (nu, xu, lu) in enumerate(validation_loader):
                    #lu[0] = gender, lu[1] = id, lu[2]=phrase, lu[3] = technique, lu[4] = vowel
                    if y_status == 'none':
                        batch_valid_loss, batch_valid_summaries = model.loss(xu.float())
                valid_loss += batch_valid_loss
                #log average validation loss for batch
                valid_loss_arr.append(valid_loss.detach().numpy())
                # scheduler.step(valid_loss)
                # # check early stopping criterion
                # if optimizer.param_groups[0]['lr'] == min_lr:
                #     ut.save_model_by_name(model, i)
                #     return

            # modify the progress bar
            if y_status == 'none':
                pbar.set_postfix(
                    loss='{:.2e}'.format(train_loss),
                    rec='{:.2e}'.format(summaries['gen/rec']),
                    kl='{:.2e}'.format(summaries['gen/kl_z']),
                    bvloss='{:.2e}'.format(valid_loss))
            elif y_status == 'semisup':
                pbar.set_postfix(
                    loss='{:.2e}'.format(loss),
                    acc='{:.2e}'.format(acc))
            elif y_status == 'fullsup':
                pbar.set_postfix(
                    loss='{:.2e}'.format(loss),
                    kl='{:.2e}'.format(summaries['gen/kl_z']))
            pbar.update(1)

            epoch += 1
            if epoch == epochs_max:
                ut.save_model_by_name(model, epoch, 
                                        train_loss_arr[:-1], valid_loss_arr,
                                        rec_loss_arr, kld_arr)
                return
