#  A lightweight GRPOTrainer converted from PPOTrainer.

#  This class assumes **group‑normalized scalar rewards** are already computed
#  externally and passed to `step()`.

#  Major differences vs. PPOTrainer
#  -------------------------------
#  - No value baseline (advantage == reward broadcast).
#  - No GAE; gamma=lam are ignored.
#  - KL penalty (beta) is applied directly in the loss term instead of being
#    included in the per‑token reward.
#  - Value‑head parameters are frozen; vf_loss is skipped.
#  - Optional whitening / scaling switches off by default.

from __future__ import annotations

import torch
from torch import Tensor
from typing import List, Optional
from trl.trainer import PPOConfig
from trl.core import (
    entropy_from_logits,
    logprobs_from_logits,
    masked_mean,
    masked_var,
    clip_by_value,
    flatten_dict
)

# re‑use the heavily patched PPOTrainer from the user as our base
from mytrl.ppo_trainer import PPOTrainer


class GRPOConfig(PPOConfig):
    """Keeps all PPOConfig fields but adds one: `beta` for KL weight."""

    def __init__(self, *args, beta: float = 0.0, loss_type: str = 'grpo', e_high : float = 0.2, e_low : float = 0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta  # KL coefficient
        # sensible GRPO defaults
        self.whiten_rewards = False
        self.vf_coef = 0.0  # disable value loss
        self.loss_type = loss_type
        if self.loss_type == 'dr_grpo':
            self.scale_rewards = False
        self.max_completion_length = 256
        self.e_high = e_high
        self.e_low = e_low


class GRPOTrainer(PPOTrainer):
    """PPOTrainer tweaked to implement GRPO semantics."""

    def __init__(self, config: GRPOConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # 1. freeze / detach value head completely
        self.config.vf_coef = 0.0
        if hasattr(self.model, "v_head"):
            for p in self.model.v_head.parameters():
                p.requires_grad_(False)

        # 2. ensure internal scaling is off (we expect pre‑scaled rewards)
        self.config.use_score_scaling = False
        self.config.use_score_norm = False
        self.running = None  # not used any more

    # ------------------------------------------------------------------
    # reward & advantage helpers
    # ------------------------------------------------------------------
    def compute_rewards(
        self,
        scores: Tensor,
        logprobs: Tensor,
        ref_logprobs: Tensor,
        masks: Tensor,
    ):
        """Return **token‑broadcasted** reward and KL (for logging only).

        • `scores` are already normalized scalar rewards (shape B)
        • We simply write them onto the last token of each sample and then
          broadcast later.
        """
        kls = (logprobs - ref_logprobs)  # plain KL, per‑token
        # --- GRPO reward broadcast ---
        reward = scores.unsqueeze(1) * masks.float()     # (B, T)
        non_score_reward = torch.zeros_like(reward)      #  Dummy
        return reward, non_score_reward, kls

    def compute_advantages(
        self,
        values: Tensor,
        rewards: Tensor,
        mask: Tensor,
    ):
        """In GRPO we directly use *reward as advantage* (no baseline)."""
        advantages = rewards.detach()
        returns = advantages  # placeholder; vf is frozen
        return values * 0.0, advantages, returns

    # ------------------------------------------------------------------
    # loss overrides
    # ------------------------------------------------------------------
    def loss(
        self,
        old_logprobs: Tensor,
        values: Tensor,
        logits: Tensor,
        vpreds: Tensor,
        logprobs: Tensor,
        mask: Tensor,
        advantages: Tensor,
        returns: Tensor,
        ref_logprobs: Tensor,
    ):
        """Policy loss identical to PPO but adds `beta*KL` term and skips VF."""
        ratio = torch.exp(logprobs - old_logprobs)
        pg_losses1 = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1.0 - self.config.e_low, 1.0 + self.config.e_high
        )

        if getattr(self.config, "loss_type", "grpo") == "dr_grpo":
            L_max   = getattr(self.config, "max_completion_length", mask.size(1))
            denom   = mask.size(0) * L_max          # B × L_max
            pg_loss = (torch.max(pg_losses1, pg_losses2) * mask.float()).sum() / denom
        else:
            pg_loss = masked_mean(torch.max(pg_losses1, pg_losses2), mask)

        # ----- DeepSeek-style KL to reference ---------------------
        if self.config.beta != 0.0:
            u       = torch.exp(ref_logprobs - logprobs)          # π_ref / π_new
            per_tok = u - (ref_logprobs - logprobs) - 1           # u - log u - 1
            
            if getattr(self.config, "loss_type", "grpo") == "dr_grpo":
                kl = (per_tok * mask.float()).sum() / denom
            else:
                kl = masked_mean(per_tok, mask)
        else:
            kl = logprobs.new_zeros(())                      # scalar 0

        # KL term (use estimator from PPO paper)
        # kl = masked_mean(old_logprobs - logprobs, mask)

        total_loss = pg_loss + self.config.beta * kl

        # ----- dummy -----
        zero = torch.tensor(0.0, device=logits.device)
        vf_loss     = zero          # ignore value loss
        vf_clipfrac = zero
        return_mean = zero
        return_var  = zero
        value_mean  = zero
        value_var   = zero
        # ----------------------

        stats = dict(
                loss=dict(policy=pg_loss.detach(),
                        value=vf_loss.detach(),
                        total=total_loss.detach()),
                policy=dict(
                    entropy=masked_mean(entropy_from_logits(logits), mask).detach(),
                    approxkl=kl.detach(),          # 편의상 approxkl = kl
                    policykl=kl.detach(),
                    clipfrac=zero,
                    advantages=advantages.detach(),
                    advantages_mean=masked_mean(advantages, mask).detach(),
                    ratio=ratio.detach(),
                ),
                returns=dict(mean=return_mean, var=return_var),
                val=dict(
                    vpred=zero,
                    error=zero,
                    clipfrac=vf_clipfrac,
                    mean=value_mean,
                    var=value_var,
                ),
        )
        return total_loss, zero, flatten_dict(stats)