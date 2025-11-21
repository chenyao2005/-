import os
import torch
import torch.nn as nn
import timm
from huggingface_hub.errors import LocalEntryNotFoundError

"""Minimal ViT model scaffold (Step 4).
Functions:
 - create_vit_model: build ViT with freeze mode.
 - summary_trainable: list trainable parameter names.
"""

def create_vit_model(
    model_name: str = "vit_base_patch16_224.augreg_in1k",
    num_classes: int = 27,
    pretrained: bool = True,
    freeze_mode: str = "full",
):
    """
    Creates a Vision Transformer (ViT) model with a custom classifier head.

    Args:
        model_name (str): Name of the ViT model to create from timm.
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained weights.
        freeze_mode (str): 'full' or 'head-only'.

    Returns:
        model (torch.nn.Module): The created model.
        trainable_params (int): Number of trainable parameters.
    """
    try:
        # Set environment variable for robust downloading
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    except Exception as e:  # Handle network-related errors gracefully
        error_str = str(e).lower()
        network_keywords = ["connect", "timeout", "network", "internet", "503", "502"]
        is_network_issue = any(keyword in error_str for keyword in network_keywords) or isinstance(e, LocalEntryNotFoundError)
        if pretrained and is_network_issue:
            print(
                f"Warning: Failed to download pretrained weights for '{model_name}' due to a network issue."
                " Falling back to randomly initialized weights (pretrained=False)."
            )
            model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        else:
            hint = (
                f"\n\n[HINT] Failed to create model '{model_name}': {e}\n"
                "If this is a network issue, please check your internet connection, firewall, or proxy settings.\n"
                "You can also try manually downloading the model weights and placing them in the timm cache.\n"
            )
            raise RuntimeError(hint) from e


    # Freeze layers
    if freeze_mode == 'full':
        print("Training mode: Full fine-tuning")
        for param in model.parameters():
            param.requires_grad = True
    elif freeze_mode == 'head_only':
        print("Training mode: Head only")
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the head
        classifier_attr = 'head' if 'vit' in model_name else 'fc'
        if hasattr(model, classifier_attr):
            for param in getattr(model, classifier_attr).parameters():
                param.requires_grad = True
        else:
            # A general approach for other models, may need adjustment
            try:
                for param in model.get_classifier().parameters():
                    param.requires_grad = True
            except AttributeError:
                raise AttributeError(
                    f"Model {model_name} does not have a standard 'head' or 'fc' attribute, and get_classifier() failed."
                    " Cannot apply 'head_only' freeze mode."
                )
    else:
        raise ValueError(f"freeze_mode must be 'full' or 'head_only', got {freeze_mode}")

    # Get a summary of the model and trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, trainable_params


def summary_trainable(model: torch.nn.Module):
    return [n for n, p in model.named_parameters() if p.requires_grad]


__all__ = ["create_vit_model", "summary_trainable"]

