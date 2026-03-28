"""
train_molora.py
───────────────
Mixture-of-LoRA-Experts (MoLoRA) fine-tuning of Qwen3-VL-8B-Instruct
on SPIQA (Test-A / Train split).

Architecture
────────────
Each targeted linear layer in the LLM (q/k/v/o/gate/up/down) is replaced
by a MoLoRALinear module that contains K independent LoRA adapters plus a
lightweight router network:

    ┌─────────────────────────────────────────────┐
    │              MoLoRALinear                    │
    │                                              │
    │  x ──► base_weight (frozen / 4-bit) ──────► │
    │   └──► router(visual_summary) ──► weights   │
    │         │                                    │
    │         ├──► LoRA_0(x) ──┐                  │
    │         ├──► LoRA_1(x) ──┼──► weighted sum  │
    │         └──► LoRA_K(x) ──┘         │        │
    │                                    ▼        │
    │                          base_out + delta ──►│
    └─────────────────────────────────────────────┘

The router is fed a *visual summary token* — the mean-pooled visual
token sequence extracted from the vision encoder output — which encodes
what kind of figure is being processed (chart, table, diagram, etc.).
This allows the model to activate different adapters for different
figure types without any explicit supervision.

Load-balancing loss
───────────────────
A standard MoE auxiliary loss penalises router collapse (all routing
weight concentrating on a single expert):

    L_aux = num_experts * sum(f_i * p_i)

where f_i = fraction of tokens routed to expert i,
      p_i = mean routing probability for expert i.

This is added to the language-modelling loss with weight `aux_loss_coef`.

W&B logging
───────────
In addition to the standard metrics from train.py, logs:
  • molora/router_entropy          — per-step routing entropy (higher = more balanced)
  • molora/expert_N_load           — fraction of examples routing to each expert
  • molora/aux_loss                — load-balancing loss value
  • molora/top_expert_dominance    — max routing weight (collapse indicator)

Usage
─────
# Single GPU, 4 experts, rank-16 per expert
python train_molora.py \\
    --num_experts 4 \\
    --molora_r 16 \\
    --aux_loss_coef 0.01 \\
    --wandb_project spiqa-molora

# Compare against vanilla LoRA baseline
python train.py --lora_r 32 --wandb_project spiqa-molora --wandb_run_name baseline-lora-r32

# Ablate number of experts
for K in 2 4 8; do
  python train_molora.py --num_experts $K --wandb_run_name molora-K$K
done
"""

import os
import json
import math
import random
import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration as AutoModelForVision2Seq,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    Trainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
"You are a study coach assessing a student's answer to a question about an image from a scientific paper.\n"
"You have provide a verdict, an error_category and a feedback after assessing the student's answer.\n\n"
"Possible verdicts:\n"
" - correct: the answer is accurate and complete\n"
" - partially correct : the answer gets some aspects right but misses or misidentifies others\n"
" - incorrect : the answer is completely wrong or contradicts the figure\n\n"
"An incorrect or partially correct student answer may have one of these error types:\n"
" - omission: Missing one or more key points in the answer\n"
" - factual: Giving factual data that contradicts the figure, or misreading it (e.g. axes, legends, trends)\n"
" - conceptual: Misunderstanding a concept or drawing a wrong conclusion from the data\n\n"
"Assess the student's response and format your output exactly as:\n"
"verdict = <incorrect | partially correct | correct>\n"
"error_category = <omission | factual | conceptual | N/A>\n"
"feedback = <concise study coach explanation, second-person tone>\n\n"
"If verdict is correct, set error_category = N/A and feedback = N/A. Do not use bullet points of bold text."
)




# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — Configuration
# ══════════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = dict(
    # ── Model ──────────────────────────────────
    model_id            = "Qwen/Qwen3-VL-8B-Instruct",
    # ── SPIQA paths ────────────────────────────
    spiqa_train_json    = "./spiqa_plus/SPIQA_plus_train_1500.json",
    spiqa_image_dir     = "./spiqa/train_val/SPIQA_train_val_Images",
    # Dedicated val split — drawn from SPIQA_val.json, sized by val_split_ratio
    spiqa_val_json      = "./spiqa_plus/SPIQA_plus_val_200.json",
    spiqa_val_image_dir = "./spiqa/train_val/SPIQA_train_val_Images",   # set to train images dir if shared
    # ── Training ───────────────────────────────
    output_dir          = "./outputs/qwen3vl-8b-spiqa-molora-2",
    num_train_epochs    = 1,
    per_device_batch    = 1,
    grad_accum_steps    = 8,
    learning_rate       = 2e-4,
    max_seq_length      = 2048,
    warmup_ratio        = 0.05,
    lr_scheduler        = "cosine",
    save_steps          = 200,
    logging_steps       = 10,
    eval_steps          = 200,
    val_split_ratio     = 0.02,
    max_samples         = 2_000,
    seed                = 42,
    # ── Image resolution ───────────────────────
    min_pixels          = 256 * 28 * 28,
    max_pixels          = 1280 * 28 * 28,
    # ── MoLoRA ─────────────────────────────────
    num_experts         = 4,        # K: number of LoRA experts per layer
    molora_r            = 16,       # rank per expert (total capacity ≈ K*r)
    molora_alpha        = 32,       # scaling: alpha / r applied to each expert output
    molora_dropout      = 0.05,
    # Target modules — subset of Qwen3-VL LLM attention + MLP projections
    # Use fewer modules than vanilla LoRA to keep memory manageable
    target_modules      = "q_proj,v_proj,o_proj,gate_proj",
    # Router
    router_hidden_dim   = 128,      # hidden size of the 2-layer router MLP
    router_noise        = 0.5,      # jitter noise during training
    # Temperature annealing: router starts at router_temp_init (near-uniform softmax)
    # and anneals linearly to 1.0 over router_temp_anneal_frac of training.
    # High initial temperature prevents premature collapse before experts differentiate.
    router_temp_init        = 10.0, # start temperature (high → near-uniform routing)
    router_temp_anneal_frac = 0.3,  # fraction of total steps to anneal over
    # Load-balancing loss coefficient
    aux_loss_coef       = 0.1,
    # Direct entropy bonus on router distribution
    router_entropy_coef = 0.02,
    # ── QLoRA ──────────────────────────────────
    use_qlora           = True,
    freeze_vision       = True,
    # ── W&B ────────────────────────────────────
    wandb_project       = "spiqa-molora-2",
    wandb_run_name      = None,
    wandb_mode          = "online",
    wandb_sample_preds  = 8,
)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — MoLoRA modules
# ══════════════════════════════════════════════════════════════════════

class LoRAAdapter(nn.Module):
    """Single LoRA adapter: down-projection A then up-projection B."""

    def __init__(self, in_features: int, out_features: int, r: int, dropout: float = 0.0):
        super().__init__()
        self.lora_A  = nn.Linear(in_features, r, bias=False)
        self.lora_B  = nn.Linear(r, out_features, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Kaiming init for A, zero init for B (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(x)))


class MoLoRARouter(nn.Module):
    """
    Lightweight 2-layer MLP router.

    Input:  visual_summary  — (B, D_vis) mean-pooled visual tokens,
                               projected to router_hidden_dim.
    Output: routing weights — (B, K) softmax probabilities over K experts.

    The router operates on the *visual* summary (not the hidden state)
    so that routing decisions are based on figure content, not the
    language prefix. This is the key design choice that makes
    MoLoRA meaningful for multimodal figure-QA.
    """

    def __init__(
        self,
        visual_dim:  int,
        num_experts: int,
        hidden_dim:  int  = 128,
        noise_std:   float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.noise_std   = noise_std
        self.temperature = temperature  # annealed externally during training

        self.net = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_experts),
        )
        # Initialise final layer near-zero so routing starts uniform
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        visual_summary: torch.Tensor,  # (B, D_vis)
        training: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        weights : (B, K)  — softmax routing weights
        logits  : (B, K)  — pre-softmax logits (for aux loss computation)
        """
        logits = self.net(visual_summary)  # (B, K)
        if training and self.noise_std > 0:
            # Noisy top-K gating (Shazeer et al., 2017) — improves exploration
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        # Temperature scaling: high T → near-uniform, T→1 → standard softmax.
        # Temperature is annealed from router_temp_init → 1.0 by the callback.
        weights = F.softmax(logits / max(self.temperature, 1e-3), dim=-1)
        return weights, logits


class MoLoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with K LoRA experts + a visual router.

    During the forward pass:
      1. Compute the base linear output (frozen weights, possibly 4-bit).
      2. Compute each expert's LoRA delta.
      3. Compute routing weights from the visual summary.
      4. Return base_out + weighted_sum(expert_deltas).

    The visual_summary is stored in a thread-local buffer set by the
    MoLoRAModel wrapper before each forward pass.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        num_experts: int,
        r:           int,
        alpha:       float,
        dropout:     float,
        router:      MoLoRARouter,
    ):
        super().__init__()
        self.base_linear = base_linear
        self.num_experts = num_experts
        self.scaling     = alpha / r
        self.router      = router  # shared router across all MoLoRALinear layers

        in_f  = base_linear.in_features
        out_f = base_linear.out_features

        self.experts = nn.ModuleList([
            LoRAAdapter(in_f, out_f, r, dropout)
            for _ in range(num_experts)
        ])

        # Slot for the visual summary — injected externally before forward()
        # Shape: (B, visual_dim)  or None if no visual context (pure text)
        self._visual_summary: Optional[torch.Tensor] = None

    def set_visual_summary(self, vs: Optional[torch.Tensor]):
        self._visual_summary = vs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Base output ──────────────────────────────────────────────
        base_out = self.base_linear(x)

        # ── Expert deltas ────────────────────────────────────────────
        expert_deltas = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )  # (B, seq, out_f, K)  or (*batch, out_f, K)

        # ── Routing ──────────────────────────────────────────────────
        if self._visual_summary is not None:
            weights, _ = self.router(
                self._visual_summary,
                training=self.training,
            )  # (B, K)
            # Reshape weights for broadcasting: (B, 1, 1, K)
            # (handles arbitrary sequence dimensions)
            for _ in range(expert_deltas.dim() - weights.dim()):
                weights = weights.unsqueeze(-2)
        else:
            # No visual context → uniform routing (pure text or padding)
            B = x.shape[0]
            weights = torch.full(
                (B, self.num_experts),
                1.0 / self.num_experts,
                device=x.device,
                dtype=x.dtype,
            )
            for _ in range(expert_deltas.dim() - weights.dim()):
                weights = weights.unsqueeze(-2)

        # Weighted combination of expert deltas
        delta = (expert_deltas * weights).sum(dim=-1)  # (B, seq, out_f)

        return base_out + self.scaling * delta


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — MoLoRA model wrapper
# ══════════════════════════════════════════════════════════════════════

class MoLoRAModel(nn.Module):
    """
    Wraps Qwen3-VL to inject MoLoRA layers and manage the visual summary
    injection + load-balancing loss accumulation.

    Key responsibilities:
      • Replace target Linear layers with MoLoRALinear
      • Extract visual summary from vision encoder output before LLM forward
      • Broadcast visual summary to all MoLoRALinear layers via hooks
      • Accumulate routing logits for aux loss computation
      • Expose a standard forward() compatible with HuggingFace Trainer
    """

    def __init__(
        self,
        base_model,
        num_experts:         int,
        r:                   int,
        alpha:               float,
        dropout:             float,
        target_modules:      list[str],
        router_hidden_dim:   int,
        router_noise:        float,
        router_temp_init:    float = 1.0,
        aux_loss_coef:       float = 0.0,
        router_entropy_coef: float = 0.0,
    ):
        super().__init__()
        self.base_model          = base_model
        self.aux_loss_coef       = aux_loss_coef
        self.router_entropy_coef = router_entropy_coef
        self.num_experts         = num_experts

        # Detect visual embedding dimension from model config
        vis_dim = self._get_visual_dim(base_model)
        log.info(f"Visual embedding dimension detected: {vis_dim}")

        # One shared router across all MoLoRA layers
        self.router = MoLoRARouter(
            visual_dim  = vis_dim,
            num_experts = num_experts,
            hidden_dim  = router_hidden_dim,
            noise_std   = router_noise,
            temperature = router_temp_init,
        )

        # Inject MoLoRA layers
        self.molora_layers: list[MoLoRALinear] = []
        self._inject_molora(base_model, target_modules, num_experts, r, alpha, dropout)
        log.info(f"Injected MoLoRA into {len(self.molora_layers)} linear layers")

        # Forward hook to extract visual tokens after vision encoder
        self._visual_summary: Optional[torch.Tensor] = None
        self._register_visual_hook()

    @staticmethod
    def _get_visual_dim(model) -> int:
        """
        Return the dimension of the visual summary vector that the hook will capture.

        The hook is placed on model.visual.merger, whose output is in LLM hidden-state
        space (e.g. 4096 for Qwen3-VL-8B), NOT the vision encoder's internal dim (1152).
        So we need the merger's OUTPUT dimension, which equals the LLM hidden size.
        """
        # Best source: merger's final Linear output features
        # Walk merger submodules and take the last Linear's out_features
        for name, module in model.named_modules():
            if "merger" in name and isinstance(module, nn.Linear):
                # Keep iterating — we want the LAST linear in the merger
                last_merger_linear = module
        try:
            dim = last_merger_linear.out_features
            log.info(f"Visual dim from merger output: {dim}")
            return dim
        except UnboundLocalError:
            pass

        # Fallback: LLM hidden size from config
        for attr in [
            "config.hidden_size",
            "model.config.hidden_size",
            "language_model.config.hidden_size",
        ]:
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return int(obj)
            except AttributeError:
                continue

        # Last resort
        log.warning("Could not detect visual dim, defaulting to 4096")
        return 4096

    def _inject_molora(
        self,
        model,
        target_modules: list[str],
        num_experts: int,
        r: int,
        alpha: float,
        dropout: float,
    ):
        """
        Walk the model graph and replace matching Linear layers with MoLoRALinear.
        Only replaces layers inside the LLM (not the vision encoder or merger).
        """
        replaced = 0
        for name, module in list(model.named_modules()):
            # Skip vision encoder and merger — only adapt the LLM
            if any(skip in name for skip in ["visual.", "merger."]):
                continue
            if not isinstance(module, nn.Linear):
                continue
            # Check if this layer's short name matches any target module
            short_name = name.split(".")[-1]
            if short_name not in target_modules:
                continue

            # Navigate to the parent module to replace the child
            parent, child_name = self._get_parent(model, name)
            if parent is None:
                continue

            molora = MoLoRALinear(
                base_linear = module,
                num_experts = num_experts,
                r           = r,
                alpha       = alpha,
                dropout     = dropout,
                router      = self.router,  # shared router
            )
            setattr(parent, child_name, molora)
            self.molora_layers.append(molora)
            replaced += 1

        log.info(f"Replaced {replaced} Linear layers with MoLoRALinear")

    @staticmethod
    def _get_parent(model: nn.Module, full_name: str):
        """Return (parent_module, child_attr_name) for a dotted module path."""
        parts  = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            if not hasattr(parent, part):
                return None, None
            parent = getattr(parent, part)
        return parent, parts[-1]

    def _register_visual_hook(self):
        """
        Register a forward hook on the vision encoder's final layer to
        capture and mean-pool the visual token sequence into a summary vector.
        """
        self._hook_call_count = 0

        def _hook(module, input, output):
            # Extract tensor from various output formats
            if hasattr(output, "last_hidden_state"):
                tokens = output.last_hidden_state
            elif isinstance(output, torch.Tensor):
                tokens = output
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                tokens = output[0]
                if not isinstance(tokens, torch.Tensor):
                    return
            else:
                return

            # Log shape on first call for diagnostics
            if self._hook_call_count == 0:
                log.info(f"Visual hook output shape: {tokens.shape} (dtype={tokens.dtype})")
            self._hook_call_count += 1

            # Handle all common shapes:
            #   (B, N, D) → mean pool over sequence dim
            #   (N, D)    → mean pool, add batch dim
            #   (B, D)    → use directly
            if tokens.dim() == 3:
                self._visual_summary = tokens.mean(dim=1).detach()   # (B, D)
            elif tokens.dim() == 2:
                # Ambiguous: could be (B, D) with B>1, or (N, D) unbatched
                # For Qwen3-VL merger: output is (N_vis, D_llm) without batch dim
                # Mean pool over first dim to get (1, D)
                self._visual_summary = tokens.mean(dim=0, keepdim=True).detach()
            elif tokens.dim() == 1:
                self._visual_summary = tokens.unsqueeze(0).detach()  # (1, D)

        self._expected_vis_dim = self._get_visual_dim(self.base_model)  # for reference only

        # Try progressively more specific module names for Qwen3-VL
        # Priority: merger input (most reliable) > visual.merger > visual > model.visual
        candidate_names = [
            "visual.merger",        # projector that maps vis tokens → LLM dim
            "model.visual.merger",
            "visual",               # full vision encoder
            "model.visual",
        ]

        target = None
        target_name = None
        for name, module in self.base_model.named_modules():
            if name in candidate_names:
                # Prefer the most specific match (merger over visual)
                if target_name is None or len(name) > len(target_name):
                    target = module
                    target_name = name

        if target is not None:
            target.register_forward_hook(_hook)
            log.info(f"Visual summary hook registered on: {target_name}")
        else:
            log.warning(
                "Could not find vision encoder module for visual summary hook. "
                "Router will use uniform weights (no visual routing)."
            )

    def _broadcast_visual_summary(self):
        """Push the current visual summary into all MoLoRALinear layers.
        We clone() here to ensure training always gets a fresh autograd-compatible
        tensor, not one captured under torch.inference_mode() by the eval callback.
        """
        vs = self._visual_summary.clone() if self._visual_summary is not None else None
        for layer in self.molora_layers:
            layer.set_visual_summary(vs)

    def forward(
        self,
        input_ids:             Optional[torch.Tensor] = None,
        attention_mask:        Optional[torch.Tensor] = None,
        pixel_values:          Optional[torch.Tensor] = None,
        image_grid_thw:        Optional[torch.Tensor] = None,
        labels:                Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass:
        1. Run base model (vision encoder fires → hook captures visual summary)
        2. Broadcast visual summary to all MoLoRA layers
        3. Compute LM loss via base model
        4. Add aux loss
        """
        # Step 1: visual hook fires automatically during base_model forward
        # Step 2: broadcast before the LLM layers process tokens
        self._broadcast_visual_summary()

        # Attach a routing logit collector to the router for this forward pass
        routing_logits_this_pass = []

        original_router_forward = self.router.forward

        def _collecting_router_forward(visual_summary, training=False):
            weights, logits = original_router_forward(visual_summary, training)
            routing_logits_this_pass.append(weights.detach())
            return weights, logits

        self.router.forward = _collecting_router_forward

        # Step 3: full model forward
        outputs = self.base_model(
            input_ids      = input_ids,
            attention_mask = attention_mask,
            pixel_values   = pixel_values,
            image_grid_thw = image_grid_thw,
            labels         = labels,
            **kwargs,
        )

        self.router.forward = original_router_forward  # restore

        lm_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0)

        # Step 4: aux loss — load-balancing + optional entropy bonus
        aux_loss = torch.tensor(0.0, device=lm_loss.device)
        if routing_logits_this_pass:
            stacked = torch.stack(routing_logits_this_pass, dim=0)  # (L, B, K)
            mean_w  = stacked.mean(dim=(0, 1))                       # (K,)

            # Switch Transformer load-balancing loss — penalises uneven expert usage
            if self.aux_loss_coef > 0:
                lb_loss  = self.num_experts * (mean_w * mean_w).sum()
                aux_loss = aux_loss + self.aux_loss_coef * lb_loss

            # Entropy bonus — directly maximises routing entropy to resist collapse.
            # This is the primary defence against the fast collapse seen with
            # structured-output tasks where LM loss gradients are very directional.
            if self.router_entropy_coef > 0:
                eps     = 1e-8
                entropy = -(mean_w * (mean_w + eps).log()).sum()  # scalar, ≥ 0
                # We *subtract* because we want to maximise entropy (minimise -entropy)
                aux_loss = aux_loss - self.router_entropy_coef * entropy

        total_loss = lm_loss + aux_loss

        # Expose routing info as an attribute for the W&B callback to read
        self._last_routing_weights = (
            torch.stack(routing_logits_this_pass, dim=0).mean(0)
            if routing_logits_this_pass else None
        )
        self._last_aux_loss = aux_loss.item()

        # Return a simple namespace-compatible object
        outputs.loss     = total_loss
        outputs.lm_loss  = lm_loss
        outputs.aux_loss = aux_loss
        return outputs

    def save_pretrained(self, path: str):
        """Save MoLoRA-specific weights (experts + router) separately."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Collect all trainable parameters
        state = {
            k: v for k, v in self.state_dict().items()
            if any(key in k for key in ["lora_A", "lora_B", "router"])
        }
        torch.save(state, path / "molora_weights.pt")

        # Save full config so evaluate_molora.py doesn't need --cfg_overrides
        cfg_out = {
            "num_experts":       self.num_experts,
            "aux_loss_coef":     self.aux_loss_coef,
            "molora_r":          self.molora_layers[0].experts[0].lora_A.out_features
                                 if self.molora_layers else 16,
            "molora_alpha":      self.molora_layers[0].scaling
                                 * (self.molora_layers[0].experts[0].lora_A.out_features)
                                 if self.molora_layers else 32,
            "router_hidden_dim": self.router.net[0].out_features,
            "target_modules":    list({
                name.split(".")[-2]           # second-to-last part is the layer name
                for name, _ in self.base_model.named_modules()
                if any(isinstance(getattr(mod, name.split(".")[-1], None), MoLoRALinear)
                       for mod in [self.base_model])
            }) or ["q_proj", "v_proj", "o_proj", "gate_proj"],
        }
        with open(path / "molora_config.json", "w") as f:
            json.dump(cfg_out, f, indent=2)

        log.info(f"MoLoRA weights saved to {path}")

    def trainable_parameters(self):
        """Return only parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def print_trainable_parameters(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        log.info(
            f"Trainable params: {trainable:,}  |  "
            f"Total params: {total:,}  |  "
            f"Trainable %: {100 * trainable / total:.2f}%"
        )
        return trainable, total


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — Custom Trainer (handles aux loss + routing metrics)
# ══════════════════════════════════════════════════════════════════════

class MoLoRATrainer(Trainer):
    """
    Subclass of HuggingFace Trainer that:
    1. Uses the total loss (lm + aux) for backprop
    2. Logs aux_loss and routing metrics separately
    3. Saves only MoLoRA adapter weights (avoids safetensors shared-tensor error
       caused by 144 MoLoRALinear layers all holding a reference to the same router)
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss    = outputs.loss
        return (loss, outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Replace the default full-model checkpoint with a lightweight save of
        only the trainable MoLoRA weights (experts + router).
        This avoids the safetensors RuntimeError about shared memory tensors,
        which occurs because all MoLoRALinear layers share one router instance.
        """
        from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        # Save only the unique trainable tensors: LoRA A/B weights + router
        # Collect de-duplicated trainable state (skip shared router copies)
        seen_data_ptrs = set()
        adapter_state = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            ptr = param.data_ptr()
            if ptr in seen_data_ptrs:
                continue
            seen_data_ptrs.add(ptr)
            # Use a short canonical name: strip the layer-specific prefix for router params
            if "router" in name:
                # All layers share the same router — save it once under a clean key
                short = "router." + ".".join(name.split("router.")[-1:])
                adapter_state[short] = param.data.cpu()
            else:
                adapter_state[name] = param.data.cpu()

        torch.save(adapter_state, os.path.join(output_dir, "molora_adapter.pt"))

        # Write molora_config.json so this checkpoint is self-contained
        # and evaluate_molora.py can load it directly without --cfg_overrides.
        cfg_out = {
            "num_experts":       model.num_experts,
            "aux_loss_coef":     model.aux_loss_coef,
            "router_entropy_coef": model.router_entropy_coef,
            "molora_r":          model.molora_layers[0].experts[0].lora_A.out_features
                                 if model.molora_layers else 16,
            "molora_alpha":      model.molora_layers[0].scaling
                                 * model.molora_layers[0].experts[0].lora_A.out_features
                                 if model.molora_layers else 32,
            "router_hidden_dim": model.router.net[0].out_features,
            "target_modules":    ["q_proj", "v_proj", "o_proj", "gate_proj"],
        }
        with open(os.path.join(output_dir, "molora_config.json"), "w") as f:
            json.dump(cfg_out, f, indent=2)

        # Save training state (step, epoch, best metric) for resumability
        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        # Track best checkpoint for load_best_model_at_end
        if metrics is not None and self.args.metric_for_best_model in metrics:
            metric_value = metrics[self.args.metric_for_best_model]
            if self.state.best_metric is None or (
                metric_value < self.state.best_metric
                if not self.args.greater_is_better
                else metric_value > self.state.best_metric
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        log.info(f"MoLoRA checkpoint saved to {output_dir}")

    def log(self, logs: dict, start_time=None):
        """Inject routing metrics into the log dict before passing to W&B."""
        model = self.model
        if hasattr(model, "_last_aux_loss"):
            logs["molora/aux_loss"] = model._last_aux_loss

        if hasattr(model, "_last_routing_weights") and model._last_routing_weights is not None:
            w = model._last_routing_weights  # (B, K)
            mean_w = w.mean(0)               # (K,)

            # Per-expert load fraction
            for i, load in enumerate(mean_w.tolist()):
                logs[f"molora/expert_{i}_load"] = load

            # Router entropy (higher = more balanced routing)
            eps = 1e-8
            entropy = -(mean_w * (mean_w + eps).log()).sum().item()
            logs["molora/router_entropy"] = entropy

            # Top-expert dominance (collapse detector)
            logs["molora/top_expert_dominance"] = mean_w.max().item()

        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — W&B setup
# ══════════════════════════════════════════════════════════════════════

def init_wandb(cfg: dict, trainable_params: int, all_params: int):
    if int(os.environ.get("LOCAL_RANK", 0)) != 0:
        return

    import wandb

    os.environ["WANDB_MODE"] = cfg["wandb_mode"]

    run_name = cfg["wandb_run_name"] or (
        f"molora-K{cfg['num_experts']}-r{cfg['molora_r']}"
        f"-aux{cfg['aux_loss_coef']}"
        f"-{'qlora' if cfg['use_qlora'] else 'bf16'}"
    )

    wandb.init(
        project = cfg["wandb_project"],
        name    = run_name,
        config  = {
            **{k: v for k, v in cfg.items() if not k.startswith("wandb_")},
            "trainable_params":       trainable_params,
            "total_params":           all_params,
            "trainable_pct":          100 * trainable_params / all_params,
            "effective_batch_size":   cfg["per_device_batch"] * cfg["grad_accum_steps"],
            "total_lora_rank":        cfg["num_experts"] * cfg["molora_r"],
        },
        tags = [
            "qwen3-vl", "spiqa", "molora",
            f"K{cfg['num_experts']}",
            f"r{cfg['molora_r']}",
        ],
    )
    log.info(f"W&B run: {wandb.run.url}")


class WandbMoLoRACallback(TrainerCallback):
    """
    W&B callback — reuses all metrics from the base train.py callback
    plus MoLoRA-specific routing visualisations.
    """

    def __init__(self, processor, val_examples, n_samples=8, max_new_tokens=128,
                 router_temp_anneal_frac=0.3):
        self.processor     = processor
        rng                = random.Random(0)
        self.qual_examples = rng.sample(val_examples, min(n_samples, len(val_examples)))
        self.max_new_tokens = max_new_tokens
        self._step_t       = None
        self._router_temp_anneal_frac = router_temp_anneal_frac

    def on_step_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        import time
        self._step_t = time.perf_counter()

        if model is not None and hasattr(model, "router"):
            max_steps = state.max_steps if state.max_steps and state.max_steps > 0 else 1
            progress = state.global_step / max_steps  # 0.0 → 1.0

            # ── Noise decay: linear from initial → 0 ─────────────────
            initial_noise = getattr(model.router, "_initial_noise_std", None)
            if initial_noise is None:
                model.router._initial_noise_std = model.router.noise_std
                initial_noise = model.router.noise_std
            model.router.noise_std = float(initial_noise) * (1.0 - progress)

            # ── Temperature annealing: high T → 1.0 over anneal_frac ──
            # High initial temperature forces near-uniform routing during
            # early training so experts can differentiate before the router
            # starts specialising. After anneal_frac of training, T=1.0.
            initial_temp = getattr(model.router, "_initial_temperature", None)
            if initial_temp is None:
                model.router._initial_temperature = model.router.temperature
                initial_temp = model.router.temperature
            anneal_frac = getattr(self, "_router_temp_anneal_frac", 0.3)
            if progress < anneal_frac:
                # Linear anneal from initial_temp → 1.0
                t = progress / anneal_frac
                model.router.temperature = initial_temp + t * (1.0 - initial_temp)
            else:
                model.router.temperature = 1.0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) != 0 or logs is None:
            return
        import time, wandb

        extra = {}

        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                extra[f"gpu/{i}/mem_allocated_gb"] = torch.cuda.memory_allocated(i) / 1e9
                extra[f"gpu/{i}/mem_reserved_gb"]  = torch.cuda.memory_reserved(i) / 1e9

        # Throughput
        if self._step_t is not None:
            elapsed = time.perf_counter() - self._step_t
            n = args.per_device_train_batch_size * args.gradient_accumulation_steps
            extra["throughput/samples_per_sec"] = n / max(elapsed, 1e-6)

        # Perplexity
        if "loss" in logs:
            try:    extra["train/perplexity"] = math.exp(logs["loss"])
            except: extra["train/perplexity"] = float("inf")

        key_map = {
            "loss":          "train/loss",
            "learning_rate": "train/learning_rate",
            "grad_norm":     "train/grad_norm",
            "eval_loss":     "eval/loss",
        }
        renamed = {key_map.get(k, k): v for k, v in logs.items()}
        if "eval/loss" in renamed:
            try:    renamed["eval/perplexity"] = math.exp(renamed["eval/loss"])
            except: renamed["eval/perplexity"] = float("inf")

        # Log current router noise std and temperature so their schedules are visible in W&B
        model = kwargs.get("model")
        if model is not None and hasattr(model, "router"):
            extra["molora/router_noise_std"]  = model.router.noise_std
            extra["molora/router_temperature"] = model.router.temperature

        wandb.log({**renamed, **extra})

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Log qualitative samples + routing distribution bar chart."""
        if int(os.environ.get("LOCAL_RANK", 0)) != 0 or model is None:
            return
        import wandb

        log.info(f"[W&B] Logging {len(self.qual_examples)} qualitative samples ...")

        table = wandb.Table(
            columns=["step", "paper_id", "question", "student", "ground_truth", "prediction", "figure",
                     "routing_weights"]
        )

        model.eval()
        with torch.inference_mode():
            for ex in self.qual_examples:
                caption_text = f"\nfigure_caption:  {ex['caption']}" if ex.get("caption") else ""
                messages = [
                    {
                        "role": "system",
                        "content": (
                           SYSTEM_PROMPT 
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": ex["image_path"]},
                            {"type": "text",  "text": f"{caption_text}\n\nquestion: {ex['question']}\n\nstudent: {ex['student']}"},
                        ],
                    },
                ]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                try:    img = Image.open(ex["image_path"]).convert("RGB")
                except: img = Image.new("RGB", (224, 224))

                inputs = self.processor(
                    text=text, images=[img], return_tensors="pt"
                ).to(model.base_model.device)

                gen_ids = model.base_model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
                new_toks = gen_ids[:, inputs["input_ids"].shape[1]:]
                pred = self.processor.batch_decode(new_toks, skip_special_tokens=True)[0].strip()

                # Routing weights for this example (if available)
                routing_str = "N/A"
                if (hasattr(model, "_last_routing_weights")
                        and model._last_routing_weights is not None):
                    w = model._last_routing_weights.mean(0).tolist()
                    routing_str = " | ".join(f"E{i}:{v:.2f}" for i, v in enumerate(w))

                ground_truth = (
                    f"verdict = {ex['verdict']}\n"
                    f"error_category = {ex['error_category']}\n"
                    f"feedback = {ex['feedback']}"
                )
                table.add_data(
                    state.global_step,
                    ex.get("paper_id", ""),
                    ex["question"],
                    ex.get("student", ""),
                    ground_truth,
                    pred,
                    wandb.Image(img),
                    routing_str,
                )

        wandb.log({"eval/qualitative_samples": table})

        # Log routing distribution as a bar chart
        if (hasattr(model, "_last_routing_weights")
                and model._last_routing_weights is not None):
            w = model._last_routing_weights.mean(0)
            K = w.shape[0]
            bar_data = [[f"Expert {i}", w[i].item()] for i in range(K)]
            wandb.log(
                {
                    "molora/routing_distribution": wandb.plot.bar(
                        wandb.Table(data=bar_data, columns=["Expert", "Mean Load"]),
                        "Expert", "Mean Load",
                        title=f"Routing Distribution @ step {state.global_step}",
                    )
                },
            )

        model.train()
        # Clear the visual summary so no inference_mode tensor persists into training
        model._visual_summary = None

    def on_train_end(self, args, state, control, **kwargs):
        if int(os.environ.get("LOCAL_RANK", 0)) != 0:
            return
        import wandb
        wandb.run.summary["best_eval_loss"]  = state.best_metric
        wandb.run.summary["best_checkpoint"] = state.best_model_checkpoint
        wandb.run.summary["total_steps"]     = state.global_step
        wandb.finish()


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — Data loading (shared with train.py)
# ══════════════════════════════════════════════════════════════════════

def _parse_spiqa_json(json_path: str, image_dir: str) -> list[dict]:
    """
    Parse a SPIQA JSON file (train or val) into a flat list of QA examples.

    Actual SPIQA schema:
    {
      "<paper_id>": {
        "all_figures": {
          "<figure_filename>": {
            "caption": "...",
            "content_type": "figure" | "table",
            "figure_type": "plot" | "schematic" | ...
          },
          ...
        },
        "qa": [
          {
            "question": "...",
            "answer":   "...",
            "reference": "<figure_filename>",   ← joins to all_figures
            ...
          },
          ...
        ]
      }
    }

    Each QA item is joined to its referenced figure to obtain the image path
    and caption. QA items whose referenced image file does not exist on disk
    are silently skipped.
    """
    with open(json_path) as f:
        raw = json.load(f)

    examples  = []
    image_dir = Path(image_dir)
    skipped   = 0

    for paper_id, paper_data in raw.items():
        all_figures = paper_data.get("all_figures", {})
        qa_list     = paper_data.get("qa", [])

        for qa in qa_list:
            ref = qa.get("reference", "")
            if not ref:
                skipped += 1
                continue

            # Image file sits directly in image_dir under its filename
            img_path = image_dir / paper_id / ref
            if not img_path.exists():
                skipped += 1
                continue

            # Look up caption from all_figures; fall back to empty string
            fig_meta = all_figures.get(ref, {})
            caption  = fig_meta.get("caption", "").strip()
            fig_type = fig_meta.get("figure_type", "unknown")

            examples.append({
                "paper_id":   paper_id,
                "fig_id":     ref,
                "image_path": str(img_path),
                "caption":    caption,
                "fig_type":   fig_type,
                "question":   qa["question"],
                "answer":     qa["answer"],
                "student" :   qa["student"],
                "verdict":     qa["verdict"],
                "error_category": qa["error_category"],
                "feedback" : qa["feedback"]
            })

    if skipped:
        log.debug(f"Skipped {skipped} QA items (missing reference or image file)")

    return examples, raw

def load_spiqa_train(json_path, image_dir, max_samples=None, seed=42):
    log.info(f"Loading SPIQA train data from {json_path}")
    examples, raw = _parse_spiqa_json(json_path, image_dir)
    log.info(f"Loaded {len(examples)} QA pairs from {len(raw)} papers")
    random.seed(seed)
    random.shuffle(examples)
    if max_samples is not None:
        examples = examples[:max_samples]
    return examples


def load_spiqa_val(json_path, image_dir, n_samples, seed=42):
    """
    Load n_samples examples from SPIQA_val.json.

    n_samples is computed in main() as int(n_train * val_split_ratio), so the
    val set size scales proportionally with the training set.
    """
    log.info(f"Loading SPIQA val data from {json_path}")
    examples, raw = _parse_spiqa_json(json_path, image_dir)
    log.info(f"Loaded {len(examples)} QA pairs from SPIQA val ({len(raw)} papers)")
    random.seed(seed)
    random.shuffle(examples)

    # ── Fail-fast: check requested count against available examples ──────
    available = len(examples)
    if n_samples > available:
        raise ValueError(
            f"Requested {n_samples} val examples (= n_train * val_split_ratio) "
            f"but SPIQA_val.json only contains {available} QA pairs "
            f"(after filtering for existing images).\n"
            f"  → Reduce val_split_ratio or max_samples so that "
            f"int(max_samples * val_split_ratio) ≤ {available}.\n"
            f"  → Current limit: max_samples ≤ {int(available / 0.02)} "
            f"at the default val_split_ratio=0.02."
        )

    n_samples = max(1, n_samples)
    examples  = examples[:n_samples]
    log.info(f"Using {len(examples)} val examples (val_split_ratio-proportional)")
    return examples


def make_conversation(example: dict) -> dict:
    """
    Store only flat string fields in the HuggingFace Dataset.
    The collator reconstructs the conversation structure at batch time.
    This avoids PyArrow's inability to handle mixed list/non-list content
    fields (string for system/assistant turns vs list-of-dicts for user turn).
    """
    return {
        "paper_id":   example.get("paper_id", ""),
        "image_path": example["image_path"],
        "caption":    example.get("caption", ""),
        "question":   example["question"],
        "answer":     example["answer"],
        "student":     example["student"],
        "verdict":     example["verdict"],
        "error_category": example["error_category"],
        "feedback":     example["feedback"]
    }


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — Collator
# ══════════════════════════════════════════════════════════════════════

class SPIQACollator:
    def __init__(self, processor, max_length=2048):
        self.processor  = processor
        self.max_length = max_length

    def __call__(self, examples):
        texts, all_images, valid = [], [], []

        for ex in examples:
            caption_text = (
                f"\nfigure_caption: {ex['caption']}" if ex.get("caption") else ""
            )
            conv = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": ex["image_path"]},
                        {"type": "text",  "text": f"{caption_text}\n\nquestion: {ex['question']}\n\nstudent: {ex['student']}"},
                    ],
                },
                {"role": "assistant", "content": f"verdict = {ex['verdict']}\nerror_category = {ex['error_category']}\nfeedback = {ex['feedback']}"},
            ]
            text = self.processor.apply_chat_template(
                conv, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)

            try:
                img = Image.open(ex["image_path"]).convert("RGB")
            except Exception:
                img = Image.new("RGB", (224, 224))
            all_images.append([img])
            valid.append(ex)

        # Process without truncation — max_length is set high enough that
        # truncation should never be needed. If a sequence somehow still
        # exceeds the limit, truncation would desync image token counts and
        # cause a hard crash, so we skip oversized examples with a warning
        # instead of silently corrupting them.
        safe_texts, safe_images = [], []
        for i, (text, imgs) in enumerate(zip(texts, all_images)):
            img_token_count = text.count("<|image_pad|>")
            tok_len = len(self.processor.tokenizer.encode(text))
            if tok_len > self.max_length:
                log.warning(
                    f"Skipping example {valid[i].get('paper_id','?')} — "
                    f"sequence length {tok_len} > max_length {self.max_length}. "
                    f"Increase max_seq_length to include it."
                )
                continue
            safe_texts.append(text)
            safe_images.append(imgs)

        if not safe_texts:
            # All examples in the batch were too long — return empty batch
            # (Trainer will skip it). Should not happen in normal operation.
            log.warning("All examples in batch exceeded max_length — returning empty batch.")
            safe_texts  = [texts[0]]
            safe_images = [all_images[0]]

        inputs = self.processor(
            text=safe_texts, images=safe_images,
            return_tensors="pt", padding=True,
            truncation=False,
        )

        labels = inputs["input_ids"].clone()
        assistant_token_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        for i, row in enumerate(labels):
            positions = (row == assistant_token_id).nonzero(as_tuple=True)[0]
            if len(positions) >= 1:
                labels[i, : positions[-1].item() + 2] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — Model builder
# ══════════════════════════════════════════════════════════════════════

def build_model_and_processor(cfg: dict):
    log.info(f"Loading processor from {cfg['model_id']}")
    processor = AutoProcessor.from_pretrained(
        cfg["model_id"],
        min_pixels = cfg["min_pixels"],
        max_pixels = cfg["max_pixels"],
    )
    processor.tokenizer.padding_side = "right"

    bnb_config = None
    if cfg["use_qlora"]:
        log.info("Using 4-bit NF4 quantisation for base model")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit             = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type      = "nf4",
            bnb_4bit_compute_dtype   = torch.bfloat16,
        )

    log.info(f"Loading base model {cfg['model_id']}")
    base_model = AutoModelForVision2Seq.from_pretrained(
        cfg["model_id"],
        quantization_config = bnb_config,
        dtype= torch.bfloat16 if not cfg["use_qlora"] else None,
        device_map          = "auto",
        trust_remote_code   = True,
    )

    # Freeze everything in the base model
    for param in base_model.parameters():
        param.requires_grad = False

    # Unfreeze merger/projector (cross-modal alignment)
    if not cfg["freeze_vision"]:
        for name, param in base_model.named_parameters():
            if "merger" in name:
                param.requires_grad = True
                log.info(f"  Unfreezing merger param: {name}")

    target_modules = [m.strip() for m in cfg["target_modules"].split(",")]
    log.info(f"MoLoRA target modules: {target_modules}")

    molora_model = MoLoRAModel(
        base_model          = base_model,
        num_experts         = cfg["num_experts"],
        r                   = cfg["molora_r"],
        alpha               = cfg["molora_alpha"],
        dropout             = cfg["molora_dropout"],
        target_modules      = target_modules,
        router_hidden_dim   = cfg["router_hidden_dim"],
        router_noise        = cfg["router_noise"],
        router_temp_init    = cfg["router_temp_init"],
        aux_loss_coef       = cfg["aux_loss_coef"],
        router_entropy_coef = cfg["router_entropy_coef"],
    )

    # Enable gradient checkpointing on the wrapped base model if supported
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Required for 4-bit QLoRA gradient flow
    if cfg["use_qlora"]:
        base_model.enable_input_require_grads()

    # Cast MoLoRA adapters + router to bfloat16 to match the base model.
    # nn.Linear weights default to float32 at init; without this cast the
    # LoRA forward (x @ lora_A.weight) fails with a dtype mismatch when x
    # is bfloat16.
    target_dtype = torch.bfloat16
    for layer in molora_model.molora_layers:
        layer.to(target_dtype)
    molora_model.router.to(target_dtype)
    log.info(f"Cast MoLoRA adapters and router to {target_dtype}")

    trainable, total = molora_model.print_trainable_parameters()
    return molora_model, processor, trainable, total


# ══════════════════════════════════════════════════════════════════════
# SECTION 9 — Main
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="MoLoRA fine-tuning of Qwen3-VL-8B on SPIQA")
    for k, v in DEFAULT_CONFIG.items():
        if isinstance(v, bool):
            p.add_argument(f"--{k}", default=v, action="store_true")
            p.add_argument(f"--no_{k}", dest=k, action="store_false")
        else:
            p.add_argument(f"--{k}", default=v, type=type(v) if v is not None else str)
    return vars(p.parse_args())


def main():
    cfg = parse_args()
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    # ── Data ───────────────────────────────────
    raw = load_spiqa_train(
        json_path   = cfg["spiqa_train_json"],
        image_dir   = cfg["spiqa_image_dir"],
        max_samples = cfg["max_samples"],
        seed        = cfg["seed"],
    )
    train_convs = [make_conversation(ex) for ex in raw]

    # Compute val size proportional to training set (mirrors old val_split_ratio)
    n_val = max(1, int(len(train_convs) * cfg["val_split_ratio"]))
    val_raw = load_spiqa_val(
        json_path  = cfg["spiqa_val_json"],
        image_dir  = cfg["spiqa_val_image_dir"],
        n_samples  = n_val,
        seed       = cfg["seed"],
    )
    val_convs = [make_conversation(ex) for ex in val_raw]

    train_ds = Dataset.from_list(train_convs)
    val_ds   = Dataset.from_list(val_convs)
    log.info(f"Train: {len(train_ds)}  |  Val: {len(val_ds)} (from SPIQA_val.json)")

    # ── Model ──────────────────────────────────
    model, processor, trainable_params, all_params = build_model_and_processor(cfg)

    # ── W&B ────────────────────────────────────
    if cfg["wandb_mode"] != "disabled":
        init_wandb(cfg, trainable_params, all_params)

    # ── Collator ───────────────────────────────
    collator = SPIQACollator(processor, max_length=cfg["max_seq_length"])

    # ── TrainingArguments ──────────────────────
    training_args = TrainingArguments(
        output_dir                   = cfg["output_dir"],
        num_train_epochs             = cfg["num_train_epochs"],
        per_device_train_batch_size  = cfg["per_device_batch"],
        per_device_eval_batch_size   = 1,
        gradient_accumulation_steps  = cfg["grad_accum_steps"],
        learning_rate                = cfg["learning_rate"],
        lr_scheduler_type            = cfg["lr_scheduler"],
        warmup_ratio                 = cfg["warmup_ratio"],
        bf16                         = True,
        gradient_checkpointing       = False,    # handled manually above
        logging_steps                = cfg["logging_steps"],
        save_steps                   = cfg["save_steps"],
        eval_steps                   = cfg["eval_steps"],
        eval_strategy          = "steps",
        save_total_limit             = 3,
        load_best_model_at_end       = True,
        metric_for_best_model        = "eval_loss",
        greater_is_better            = False,
        remove_unused_columns        = False,
        dataloader_num_workers       = 4,
        report_to                    = "wandb" if cfg["wandb_mode"] != "disabled" else "none",
        run_name                     = cfg.get("wandb_run_name"),
        seed                         = cfg["seed"],
    )

    # ── Callback ───────────────────────────────
    val_raw = list(val_convs)   # already flat dicts with paper_id, image_path, etc.
    cb = WandbMoLoRACallback(
        processor               = processor,
        val_examples            = val_raw,
        n_samples               = cfg["wandb_sample_preds"],
        router_temp_anneal_frac = cfg["router_temp_anneal_frac"],
    )

    # ── Trainer ────────────────────────────────
    trainer = MoLoRATrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = collator,
        callbacks     = [cb] if cfg["wandb_mode"] != "disabled" else [],
    )

    log.info("Starting MoLoRA training ...")
    trainer.train()

    # ── Save ───────────────────────────────────
    save_path = Path(cfg["output_dir"]) / "final_molora"
    log.info(f"Saving MoLoRA weights to {save_path}")
    model.save_pretrained(str(save_path))
    processor.save_pretrained(str(save_path))
    log.info("Done.")


if __name__ == "__main__":
    main()
