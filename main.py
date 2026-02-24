from techniques.prunning import PrunningStructured, PrunningUnstructured
import utils
import torch
import torch.nn as nn
import torch.optim as optim
from models.resnet_s import resnet20
from cifar10 import trainloader, testloader, trainloader_subset

def main():
    print("Start")

    checkpoint = torch.load("./resnet20_best.pth")
    model = resnet20()
    model.load_state_dict(checkpoint)
    model.eval()

    cfg = utils.ConfigModel(
        # basic info
        label="rn20a1",
        model=model,
        train_data=trainloader_subset,
        test_data=testloader,
        project_name = "imt_efficient_dl",
        path_backup = "./melyssa",

        # hyperparameters
        num_epochs=20,
        learning_rate=0.0001,
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

        # pruning settings
        pruning_method="combined",  # "unstructured", "structured", or "combined"
        structured_ratios=[0.25],  # structured pruning ratios
        unstructured_ratios=[0.7],  # unstructured pruning ratios
        avoid_overlap=True,  # avoid overlapping pruning

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

    pruneUnstructured = PrunningUnstructured(
        cfgModel=cfg,
        ratios=[0.25],
        params_to_prune=[
            (module, "weight")
            for module in cfg.model.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ],
    )

    pruneStructured = PrunningStructured(
        cfgModel=cfg,
        ratios=[0.25],  # structured ratios
        params_to_prune=[
            (module, "weight")
            for module in cfg.model.modules()
            if isinstance(module, (nn.Conv2d, nn.Linear))
        ],
    )

    score = cfg.calculate_score(
        p_s=cfg.structured_ratios[0],  # structured pruning ratio
        p_u=cfg.unstructured_ratios[0],  # unstructured pruning ratio
        q_w=16,    # quantization for weights (example value)
        q_a=16,    # quantization for activations (example value)
        w=sum(p.numel() for p in cfg.model.parameters()),  # total number of weights
        f=40.55e6,  # total number of MACs (example value for ResNet20 on CIFAR-10)
    )

    # Only unstructured pruning
    # pruneUnstructured.unstructured()

    # Only structured pruning
    # pruneStructured.structured()

    # Combined pruning with different ratios
    pruneStructured.combined(
        structured_ratios=cfg.structured_ratios,
        unstructured_ratios=cfg.unstructured_ratios,
        avoid_overlap=cfg.avoid_overlap,  # avoid overlapping pruning
    )

    cfg.train_loop()
    print("End")

if __name__ == '__main__':
    main()
