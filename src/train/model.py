import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class MultiClassSegFormer(nn.Module):
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-ade-512-512", num_classes=3):
        super(MultiClassSegFormer, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        return self.model(x).logits
