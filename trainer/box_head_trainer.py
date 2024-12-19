
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
from model import BoxHead

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

path = {}
path["images"] = "./data/hw3_mycocodata_img_comp_zlib.h5" 
path["labels"] = "./data/hw3_mycocodata_labels_comp_zlib.npy"
path["bboxes"] = "./data/hw3_mycocodata_bboxes_comp_zlib.npy"
path["masks"]  = "./data/hw3_mycocodata_mask_comp_zlib.h5"

torch.manual_seed(42)
dataset = BuildDataset(path)
full_size = len(dataset)
train_size = int(full_size * 0.8)
test_size = full_size - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 4
train_build_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
train_loader = train_build_loader.loader()
test_build_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = test_build_loader.loader()

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


model_bh = BoxHead()

epochs = 40

tb_logger = pl_loggers.TensorBoardLogger("tb_logs", name="faster_rcnn")
trainer = pl.Trainer(accelerator='gpu', devices=[2], max_epochs=epochs, logger=tb_logger, callbacks=[val_checkpoint_callback,train_checkpoint_callback])

trainer.fit(model_bh, train_loader, test_loader)

PATH = 'model_trained_boxhead.pth'
torch.save(model_bh.state_dict(), PATH)

tot_train_losses = model_bh.train_loss_epoch
plt.plot(tot_train_losses)
plt.title("Total Training loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

tot_losses = model_bh.train_class_loss_epoch
plt.plot(tot_losses)
plt.title("Train Classifier loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

tot_losses = model_bh.train_reg_loss_epoch
tot_losses = np.array([each_loss for each_loss in tot_losses])
plt.plot(tot_losses)
plt.title("Train Regressor loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

tot_val_losses = model_bh.val_loss_epoch
total_val_losses = [each_loss.cpu().numpy() for each_loss in tot_val_losses]
plt.plot(total_val_losses[1:])
plt.title("Total Validation loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

tot_losses = model_bh.val_class_loss_epoch
plt.plot(tot_losses[1:])
plt.title("Val Classifier loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

tot_losses = model_bh.val_reg_loss_epoch
tot_losses = np.array([each_loss for each_loss in tot_losses])
plt.plot(tot_losses[1:])
plt.title("Val Regressor loss per epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()