"""Three ways to flag a defective part, each returning one anomaly score per image.

1. ``patchcore_scores``: the industrial baseline. Pretrained CNN patch features of defect-free
   images form a memory bank; an image's score is how far its least normal patch is from it.
   Needs no defect examples at all (PatchCore, Roth et al., CVPR 2022, simplified).
2. ``finetune_scores``: transfer learning. An ImageNet ResNet-18 is fine-tuned to tell defective
   from defect-free, using only the few labeled shots plus defect-free training images.
3. ``vlm_scores``: a vision-language model asked whether the part is defective, scored by the
   probability it answers "Yes"; optionally shown a defect-free reference image first.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image as PILImage
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_rgb(root: Path, rel_path: str) -> PILImage.Image:
    return PILImage.open(root / rel_path).convert("RGB")


def _tensor(images: Sequence[PILImage.Image], size: int = 224) -> torch.Tensor:
    from torchvision import transforms

    tf = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return torch.stack([tf(im) for im in images])


# --- 1. PatchCore-style anomaly detection --------------------------------------------------


class _PatchFeatures(nn.Module):
    """Wide-ResNet-50 layer2 + layer3 features, layer3 upsampled, 3x3 average-pooled."""

    def __init__(self):
        super().__init__()
        from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2

        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval()
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f3 = nn.functional.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        feats = nn.functional.avg_pool2d(torch.cat([f2, f3], dim=1), 3, stride=1, padding=1)
        return feats.permute(0, 2, 3, 1).flatten(1, 2)  # (images, patches, channels)


def patchcore_scores(
    train_good: Sequence[PILImage.Image],
    test: Sequence[PILImage.Image],
    bank_fraction: float = 1.0,
    seed: int = 0,
    batch: int = 16,
) -> np.ndarray:
    device = _device()
    model = _PatchFeatures().to(device)

    def features(images):
        return torch.cat(
            [model(_tensor(images[i : i + batch]).to(device)) for i in range(0, len(images), batch)]
        )

    bank = features(list(train_good)).flatten(0, 1)
    # Full memory bank by default; PatchCore's greedy coreset only makes it smaller and faster.
    generator = torch.Generator().manual_seed(seed)
    keep = torch.randperm(len(bank), generator=generator)[: int(bank_fraction * len(bank))]
    bank = bank[keep.to(bank.device)]
    scores = []
    for start in range(0, len(test), batch):
        patches = model(_tensor(test[start : start + batch]).to(device))
        for image_patches in patches:
            nearest = torch.cdist(image_patches, bank).min(dim=1).values
            scores.append(nearest.max().item())
    return np.array(scores)


# --- 2. Transfer learning ------------------------------------------------------------------


def finetune_scores(
    train_images: Sequence[PILImage.Image],
    train_labels: Sequence[int],
    test: Sequence[PILImage.Image],
    epochs: int = 15,
    lr: float = 1e-4,
    seed: int = 0,
) -> np.ndarray:
    """Fine-tune all layers of an ImageNet ResNet-18 with a new 1-logit head; class-weighted loss
    because defect-free images outnumber the few labeled defects."""
    from torchvision.models import ResNet18_Weights, resnet18

    torch.manual_seed(seed)
    device = _device()
    net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Linear(net.fc.in_features, 1)
    net = net.to(device)
    x = _tensor(train_images)
    y = torch.tensor(train_labels, dtype=torch.float32)
    positives = max(1.0, float(y.sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((len(y) - positives) / positives))
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(y), generator=generator).split(16):
            # No flip augmentation: an upside-down metal nut ("flip") is itself a defect type.
            optimizer.zero_grad()
            loss_fn(net(x[idx].to(device)).squeeze(1), y[idx].to(device)).backward()
            optimizer.step()
    net.eval()
    with torch.no_grad():
        logits = torch.cat(
            [
                net(_tensor(test[i : i + 32]).to(device)).squeeze(1).cpu()
                for i in range(0, len(test), 32)
            ]
        )
    return logits.numpy()


# --- 3. Vision-language model --------------------------------------------------------------

VLM_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def describe(part: str) -> str:
    return part.replace("_", " ")


def vlm_prompt(part: str, with_reference: bool) -> str:
    if with_reference:
        return (
            f"The first image shows a defect-free {describe(part)} from a production line. "
            f"Is the {describe(part)} in the second image defective (for example scratched, "
            "bent, cracked, contaminated, misplaced or missing a part)? Answer Yes or No."
        )
    return (
        f"This is a photo of a {describe(part)} from a production line. Is it defective "
        "(for example scratched, bent, cracked, contaminated, misplaced or missing a part)? "
        "Answer Yes or No."
    )


VLM_SIZE = 448  # pixels; a multiple of 28, so every image is the same 16x16 = 256 visual tokens


class VLMInspector:
    """Scores P("Yes") for the question above with a local Qwen2.5-VL model."""

    def __init__(self, model_id: str = VLM_ID):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.float16, device_map={"": 0}, attn_implementation="sdpa"
        ).eval()
        tok = self.processor.tokenizer
        variants = {"yes": ["Yes", "yes", " Yes"], "no": ["No", "no", " No"]}
        self.ids = {}
        for answer, words in variants.items():
            ids = [tok.encode(w, add_special_tokens=False) for w in words]
            assert all(len(i) == 1 for i in ids), f"multi-token answer variant: {words}"
            self.ids[answer] = sorted({i[0] for i in ids})

    @torch.no_grad()
    def score(
        self, image: PILImage.Image, part: str, reference: PILImage.Image | None = None
    ) -> tuple[float, float]:
        """(P(Yes | Yes or No), P(Yes or No)): the second number shows whether the model was
        answering the question at all (low mass means it wanted to say something else)."""
        size = (VLM_SIZE, VLM_SIZE)
        images = [image.resize(size, PILImage.BICUBIC)]
        if reference is not None:
            images.insert(0, reference.resize(size, PILImage.BICUBIC))
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": vlm_prompt(part, reference is not None)})
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], add_generation_prompt=True
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(0)
        logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite VLM logits; rerun in float32 or bfloat16")
        logprobs = torch.log_softmax(logits, dim=-1)
        yes = torch.logsumexp(logprobs[self.ids["yes"]], dim=0)
        no = torch.logsumexp(logprobs[self.ids["no"]], dim=0)
        p_yes = torch.softmax(torch.stack([yes, no]), dim=0)[0].item()
        return p_yes, float(torch.exp(torch.logaddexp(yes, no)))
