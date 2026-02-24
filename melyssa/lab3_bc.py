import utils
import torch.nn as nn
import torch
import torch.optim as optim
from models.binaryconnect import BC
from models.resnet_s import resnet20
from cifar10 import trainloader, testloader, trainloader_subset
from utils import ConfigModel
def main():
    print("Start")
    print("==========")
    print("BinaryConnect")
    print("==========")

    model = BC(resnet20())

    cfg = ConfigModel(
        # basic info
        label="lab3_a1",
        model=model,
        train_data=trainloader_subset,
        test_data=testloader,
        project_name = "imt_efficient_dl",
        path_backup = "./melyssa",
        input_dtype = 'binary',
        wand_on=False,
        batch_size=128,

        # hyperparameters
        num_epochs=2,
        learning_rate=0.01,
        weight_decay=1e-3,
        optimizer_name='SGD',
        momentum=0.9,
        nesterov=True,
        label_smoothing=0.1,

        # training settings
        warmup_epochs=5,
        scheduler_name="LinearLR+CosineAnnealingLR",
        early_stopping_patience=150,
        early_stopping_min_delta=0.0,

    )
    
    cfg.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    cfg.optimizer = optim.SGD(
        cfg.model.parameters(),
        lr=cfg.learning_rate,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=cfg.nesterov,
    )
    warmup = optim.lr_scheduler.LinearLR(
        cfg.optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=cfg.warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        cfg.optimizer,
        T_max=max(1, cfg.num_epochs - cfg.warmup_epochs),
    )
    
    cfg.scheduler = optim.lr_scheduler.SequentialLR(
        cfg.optimizer,
        schedulers=[warmup, cosine],
        milestones=[cfg.warmup_epochs],
    )

    cfg.train_loop()
    
    print("End")

if __name__ == '__main__':
    main()
