#!/usr/bin/env python3
"""
Exporta um modelo BC + ativacao fake-quant (ex.: INT4) para ONNX
e compara a acuracia entre PyTorch e ONNXRuntime.

Fluxo:
1) Carrega checkpoint treinado no wrapper BC.
2) Avalia referencia PyTorch com binarizacao de pesos em runtime.
3) "Congela" pesos binarios para inferencia (sem restore) e exporta ONNX.
4) Avalia o ONNX exportado e reporta delta de acuracia.
"""

import argparse
import copy
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from cifar10 import testloader
from models.binaryconnect import BC
from models.resnet_s import resnet20


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint invalido em {path}. Esperado dict/state_dict.")

    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            cleaned[key[len("module."):]] = value
        else:
            cleaned[key] = value
    return cleaned


def _align_prefix_to_model(
    state_dict: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    model_has_model_prefix = any(k.startswith("model.") for k in model.state_dict().keys())
    ckpt_has_model_prefix = any(k.startswith("model.") for k in state_dict.keys())

    if model_has_model_prefix and not ckpt_has_model_prefix:
        return {f"model.{k}": v for k, v in state_dict.items()}
    return state_dict


def evaluate_bc_runtime(
    model_bc: BC,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float]:
    model_bc.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            if max_batches and batch_idx >= max_batches:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            model_bc.binarization()
            outputs = model_bc(inputs)
            loss = criterion(outputs, labels)
            model_bc.restore()

            running_loss += loss.item()
            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    num_batches = (batch_idx + 1) if total > 0 else 1
    return running_loss / num_batches, 100.0 * correct / max(1, total)


def evaluate_plain_torch(
    model: torch.nn.Module,
    device: torch.device,
    max_batches: int = 0,
) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            if max_batches and batch_idx >= max_batches:
                break

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            pred = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()

    num_batches = (batch_idx + 1) if total > 0 else 1
    return running_loss / num_batches, 100.0 * correct / max(1, total)


def export_onnx(
    model: torch.nn.Module,
    onnx_path: str,
    device: torch.device,
    opset: int = 17,
):
    model.eval()
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    dummy = torch.randn(1, 3, 32, 32, device=device)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )


def evaluate_onnx(
    onnx_path: str,
    max_batches: int = 0,
) -> Tuple[float, float]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime nao encontrado. Instale com: pip install onnxruntime"
        ) from exc

    providers = []
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess = ort.InferenceSession(onnx_path, providers=providers)
    input_name = sess.get_inputs()[0].name

    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(testloader):
            if max_batches and batch_idx >= max_batches:
                break

            outputs = sess.run(None, {input_name: inputs.numpy().astype(np.float32)})[0]
            outputs_t = torch.from_numpy(outputs)
            labels_t = labels

            loss = criterion(outputs_t, labels_t)
            pred = outputs_t.argmax(dim=1)

            running_loss += loss.item()
            total += labels_t.size(0)
            correct += (pred == labels_t).sum().item()

    num_batches = (batch_idx + 1) if total > 0 else 1
    return running_loss / num_batches, 100.0 * correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Exporta BC+INT4 para ONNX e avalia.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./final_model_rn20a1.pth",
        help="Checkpoint .pth treinado com BC.",
    )
    parser.add_argument(
        "--onnx-out",
        type=str,
        default="./exports/final_model_rn20a1_bc_int4.onnx",
        help="Caminho do arquivo ONNX exportado.",
    )
    parser.add_argument(
        "--binarized-pth-out",
        type=str,
        default="./exports/final_model_rn20a1_binarized_frozen.pth",
        help="Checkpoint com pesos congelados em {-1,+1,0} para inferencia.",
    )
    parser.add_argument(
        "--activation-bits",
        type=int,
        default=4,
        help="Bits de ativacao fake-quant no wrapper BC.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="Opset ONNX de exportacao.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Limita batches para teste rapido (0 = conjunto completo).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Forca execucao em CPU.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    model_bc = BC(
        resnet20(),
        quantize_activations_int8=True,
        activation_bits=args.activation_bits,
    ).to(device)

    state_dict = _load_state_dict(args.checkpoint)
    state_dict = _align_prefix_to_model(state_dict, model_bc)
    missing, unexpected = model_bc.load_state_dict(state_dict, strict=False)
    print(f"Load state_dict -> missing: {len(missing)} | unexpected: {len(unexpected)}")

    torch_bc_loss, torch_bc_acc = evaluate_bc_runtime(
        model_bc=model_bc,
        device=device,
        max_batches=args.max_batches,
    )
    print(f"PyTorch BC runtime -> loss: {torch_bc_loss:.4f} | acc: {torch_bc_acc:.2f}%")

    # Congela pesos binarios no modelo interno para exportacao de inferencia.
    export_bc = copy.deepcopy(model_bc).to(device)
    export_bc.eval()
    with torch.no_grad():
        export_bc.binarization()
    frozen_model = export_bc.model
    frozen_model.eval()

    os.makedirs(os.path.dirname(args.binarized_pth_out) or ".", exist_ok=True)
    torch.save(frozen_model.state_dict(), args.binarized_pth_out)
    print(f"Checkpoint binarizado congelado salvo em: {args.binarized_pth_out}")

    frozen_loss, frozen_acc = evaluate_plain_torch(
        model=frozen_model,
        device=device,
        max_batches=args.max_batches,
    )
    print(f"PyTorch frozen model -> loss: {frozen_loss:.4f} | acc: {frozen_acc:.2f}%")

    export_onnx(
        model=frozen_model,
        onnx_path=args.onnx_out,
        device=device,
        opset=args.opset,
    )
    print(f"ONNX exportado em: {args.onnx_out}")

    onnx_loss, onnx_acc = evaluate_onnx(
        onnx_path=args.onnx_out,
        max_batches=args.max_batches,
    )
    print(f"ONNXRuntime -> loss: {onnx_loss:.4f} | acc: {onnx_acc:.2f}%")

    print("\nResumo de deltas:")
    print(f"- Delta (frozen - BC runtime): {frozen_acc - torch_bc_acc:+.3f} pp")
    print(f"- Delta (ONNX - frozen): {onnx_acc - frozen_acc:+.3f} pp")
    print(f"- Delta (ONNX - BC runtime): {onnx_acc - torch_bc_acc:+.3f} pp")


if __name__ == "__main__":
    main()
