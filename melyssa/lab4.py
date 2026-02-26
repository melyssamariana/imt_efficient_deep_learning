import utils
import torch.nn as nn
import torch
import torch.optim as optim
from models.resnet import ResNet18
from cifar10 import trainloader, testloader, trainloader_subset
from techniques.prunning_uns_global_fp16 import Prunning
from utils import ConfigModel
def main():
    print("Start")

    checkpoint = torch.load("melyssa/best_model_lab1_a1.pth")
    model = ResNet18()
    model.load_state_dict(checkpoint)
    model.eval()

    #Convert to FP16
    model.half()
    input_dtype = torch.float16
    precision_notes = "FP16 (model + inputs)"

    cfg = ConfigModel(
        # basic info
        label="lab4_a1",
        model=model,
        train_data=trainloader_subset,
        test_data=testloader,
        project_name = "imt_efficient_dl",
        path_backup = "./melyssa",
        input_dtype = torch.float16,
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

    prune = Prunning(
        cfgModel=cfg,
        pruning_percentage=0.5,
        ratios=[0.0, 0.25 ],#, 0.5, 0.75, 0.9],
        params_to_prune=[
            (module, "weight") for module in cfg.model.modules() if isinstance(module, (nn.Conv2d, nn.Linear))
        ],
    )

    prune.unstructured()

    #cfg.train_loop()
    
    print("End")

if __name__ == '__main__':
    main()
