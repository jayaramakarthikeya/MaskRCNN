

import torch
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.utils import draw_bounding_boxes
from PIL import ImageColor
import pytorch_lightning as pl
from utils import pretrained_models_680
import torch.nn as nn
from utils import MultiApply, conv_box_to_corners, conv_box_to_xywh, output_decoding_postprocess
from torchvision.models.detection.image_list import ImageList
from datasets import BuildDataLoader, BuildDataset
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.loggers import tensorboard
from pytorch_lightning.callbacks import EarlyStopping
from model import RPNHead

path = {}
path["images"] = "./MaskRCNN/data/hw3_mycocodata_img_comp_zlib.h5" 
path["labels"] = "./MaskRCNN/data/hw3_mycocodata_labels_comp_zlib.npy"
path["bboxes"] = "./MaskRCNN/data/hw3_mycocodata_bboxes_comp_zlib.npy"
path["masks"] = "./MaskRCNN/data/hw3_mycocodata_mask_comp_zlib.h5"
# load the data into data.Dataset
torch.manual_seed(42)
dataset = BuildDataset(path)


# build the dataloader
# set 20% of the dataset as the training data
full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size


# random split the dataset into training and testset

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 4
train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()

early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=3, verbose=True, mode="min")

val_checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./training_data_new_model_1",
    filename="val_loss{epoch:02d}-{val_loss:.2f}",
    save_top_k=3,
    mode="min",
)
train_checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",
    dirpath="./training_data_new_model_1",
    filename="train_loss{epoch:02d}-{train_loss:.2f}",
    save_top_k=3,
    mode="min",
)

model = RPNHead(device="cuda:2",batch_size=batch_size)

epochs = 40

tb_logger = pl_loggers.TensorBoardLogger("tb_logs", name="faster_rcnn")
trainer = pl.Trainer(accelerator='gpu', devices=[2], max_epochs=epochs, logger=tb_logger, callbacks=[early_stop_callback,val_checkpoint_callback,train_checkpoint_callback]) # type: ignore

trainer.fit(model, train_loader, test_loader)

PATH = 'model_rpn.pth'
torch.save(model.state_dict(), PATH)

tot_train_losses = model.train_loss_epoch
plt.plot(tot_train_losses)
plt.title("Total Training loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("train_loss.png")

tot_losses = model.train_class_loss_epoch
plt.plot(tot_losses)
plt.title("Train Classifier loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("train_class_loss.png")

tot_losses = model.train_reg_loss_epoch
plt.plot(tot_losses)
plt.title("Train Regressor loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("train_reg_loss.png")

tot_losses = model.val_loss_epoch
plt.plot(tot_losses)
plt.title("Validation loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("val_loss.png")

tot_losses = model.val_class_loss_epoch
plt.plot(tot_losses)
plt.title("Validation Classifier loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("val_class_loss.png")

tot_losses = model.val_reg_loss_epoch
plt.plot(tot_losses)
plt.title("Validation Regressor loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("val_reg_loss.png")