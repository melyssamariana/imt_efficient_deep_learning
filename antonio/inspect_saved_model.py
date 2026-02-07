import argparse
import os
import torch

from models.resnet import ResNet18, ResNet34, ResNet50


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    if isinstance(checkpoint, dict):
        return checkpoint
    return None


def _state_dict_size_bytes(state_dict):
    total = 0
    for tensor in state_dict.values():
        if hasattr(tensor, "numel") and hasattr(tensor, "element_size"):
            total += tensor.numel() * tensor.element_size()
    return total


def _try_load(model_fn, state_dict):
    model = model_fn()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    is_full_match = len(missing) == 0 and len(unexpected) == 0
    return model, is_full_match, missing, unexpected


def _format_count(n):
    return f"{n:,}"


def main():
    parser = argparse.ArgumentParser(description="Inspect a saved PyTorch model.")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="saved_models/final_model_89.pth",
        help="Path to the saved model (.pth).",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    if state_dict is None:
        raise ValueError("Could not extract a state_dict from the checkpoint.")

    file_size_bytes = os.path.getsize(checkpoint_path)
    tensor_bytes = _state_dict_size_bytes(state_dict)

    print("Checkpoint summary")
    print("-------------------")
    print(f"Path: {checkpoint_path}")
    print(f"File size: {file_size_bytes / (1024 ** 2):.2f} MB")
    print(f"State dict keys: {len(state_dict)}")
    print(f"State dict tensor bytes: {tensor_bytes / (1024 ** 2):.2f} MB")

    total_params = sum(t.numel() for t in state_dict.values() if hasattr(t, "numel"))
    print(f"Total parameters (from tensors): {_format_count(total_params)}")

    candidates = [
        ("ResNet18", ResNet18),
        ("ResNet34", ResNet34),
        ("ResNet50", ResNet50),
    ]

    best_match = None
    for name, fn in candidates:
        model, is_full_match, missing, unexpected = _try_load(fn, state_dict)
        match_info = {
            "name": name,
            "model": model,
            "full_match": is_full_match,
            "missing": missing,
            "unexpected": unexpected,
        }
        if is_full_match:
            best_match = match_info
            break
        if best_match is None or (len(missing) + len(unexpected)) < (
            len(best_match["missing"]) + len(best_match["unexpected"])
        ):
            best_match = match_info

    if best_match is not None:
        model = best_match["model"]
        model_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\nArchitecture probe")
        print("-------------------")
        print(f"Best match: {best_match['name']}")
        print(f"Full match: {best_match['full_match']}")
        print(f"Model parameters: {_format_count(model_params)}")
        print(f"Trainable parameters: {_format_count(trainable_params)}")

        if not best_match["full_match"]:
            print(f"Missing keys: {len(best_match['missing'])}")
            print(f"Unexpected keys: {len(best_match['unexpected'])}")


if __name__ == "__main__":
    main()
