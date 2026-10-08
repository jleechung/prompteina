# Modified by the Prompteina authors in 2026.
# Changes are licensed under Apache 2.0; see LICENSE and NOTICE.
import copy
import os
import random
from functools import partial
from typing import Literal

import lightning as L
import torch
from jaxtyping import Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger
from omegaconf import OmegaConf

from proteinfoundation.flow_matching.product_space_flow_matcher import ProductSpaceFlowMatcher
from proteinfoundation.nn.genie2 import Genie2Denoiser
from proteinfoundation.utils.file_utils import create_dir as _create_dir
from proteinfoundation.utils.sample_utils import add_clean_samples, sample_formatting
from proteinfoundation.utils.training_handlers import handle_batch_conditioning
from proteinfoundation.utils.validation_utils import clean_validation_files

# Architecture selection: v1 is the default.
# Set USE_V2_COMPLEXA_ARCH=True in .env to use v2 (ligand / AME models).
_USE_V2 = os.getenv("USE_V2_COMPLEXA_ARCH", "False") == "True"
if _USE_V2:
    from proteinfoundation.nn.local_latents_transformer_v2 import (
        LocalLatentsTransformer,
    )
else:
    from proteinfoundation.nn.local_latents_transformer import (
        LocalLatentsTransformer,
    )

from proteinfoundation.nn.modules.text_encoder import TextEncoder
from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3
from proteinfoundation.partial_autoencoder.autoencoder import AutoEncoder
from proteinfoundation.rewards.reward_utils import compute_reward_from_samples, initialize_reward_model
from proteinfoundation.search.search_factory import instantiate_refinement, instantiate_search
from proteinfoundation.search.search_utils import (
    append_samples,
    clone_sample_dict,
    combine_lookahead_and_final,
    expand_hotspot_mask,
)
from proteinfoundation.utils.fold_utils import extract_cath_code_from_batch
from proteinfoundation.utils.pdb_utils import write_prot_to_pdb

create_dir = rank_zero_only(_create_dir)


class Proteina(L.LightningModule):
    def __init__(self, cfg_exp, store_dir=None, autoencoder_ckpt_path=None):
        super().__init__()
        if not os.environ.get("USE_V2_COMPLEXA_ARCH"):
            logger.info("USE_V2_COMPLEXA_ARCH not set, using v1 architecture (default)")
        self.save_hyperparameters()
        self.cfg_exp = cfg_exp
        self.inf_cfg = None  # Only for inference runs
        self.validation_output_lens = {}
        self.validation_output_data = []
        self.store_dir = store_dir if store_dir is not None else "./tmp"
        self.val_path_tmp = os.path.join(self.store_dir, "val_samples")
        # create_dir(self.val_path_tmp)

        if "local_latents" in cfg_exp.product_flowmatcher:
            if autoencoder_ckpt_path is not None:
                # Allow adding new keys
                logger.info(f"Manually setting autoencoder_ckpt_path to {autoencoder_ckpt_path}")
                OmegaConf.set_struct(cfg_exp, False)
                # Update the configuration with the new key-value pair
                cfg_exp.autoencoder_ckpt_path = autoencoder_ckpt_path
                # Re-enable struct mode if needed
                OmegaConf.set_struct(cfg_exp, True)

            self.autoencoder, self.latent_dim = self.load_autoencoder(cfg_exp, freeze_params=True)
            # Add right latent dimensionality in the config file, needed to instantiate the flow matcher below
            if self.autoencoder is not None:
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim

        self.fm = ProductSpaceFlowMatcher(cfg_exp)

        # Neural network
        if cfg_exp.nn.name == "ca_af3":
            self.nn = ProteinTransformerAF3(**cfg_exp.nn)
        # elif cfg_exp.nn.name == "ca_af3_int":
        #     self.nn = ProteinTransformerAF3Int(**cfg_exp.nn)
        elif cfg_exp.nn.name == "local_latents_transformer":
            self.nn = LocalLatentsTransformer(**cfg_exp.nn, latent_dim=self.latent_dim)
        # elif cfg_exp.nn.name == "local_latents_transformer_int":
        #     self.nn = LocalLatentsTransformerInt(
        #         **cfg_exp.nn, latent_dim=self.latent_dim
        #     )
        elif cfg_exp.nn.name == "ca_genie2":
            self.nn = Genie2Denoiser(**cfg_exp.nn)
        else:
            raise OSError(f"Wrong nn selected for CAFlow {cfg_exp.nn.name}")

        # Text conditioning
        self.text_encoder = None
        self._text_cond_enabled = False
        text_cond_cfg = getattr(cfg_exp, "text_cond", None)
        if text_cond_cfg is not None and getattr(text_cond_cfg, "enabled", False):
            self._text_cond_enabled = True
            self.text_encoder = TextEncoder(
                text_cond_cfg,
                dim_cond=cfg_exp.nn.dim_cond,
                token_dim=cfg_exp.nn.token_dim,
            )

        # Model Guidance (Tang et al., 2024)
        # Fact-level CFG dropout: randomly zero token embeddings for one
        # alignment-annotated fact per sample. Teaches the model which text
        # facts drive which structural features, making CFG fact-specific.
        self._p_fact_drop = float(getattr(text_cond_cfg, "p_fact_drop", 0.0)) if text_cond_cfg is not None else 0.0

        # Diffusion loss weight (1.0 = normal; 0.0 = alignment-only training)
        loss_cfg = getattr(cfg_exp, "loss", None)
        self._diff_loss_weight = float(getattr(loss_cfg, "diff_loss_weight", 1.0))

        # Output distillation to parent (regularization for alignment training).
        # Swaps in frozen parent weights, runs a second forward pass to get
        # parent outputs, then penalizes MSE between model and parent outputs.
        distill_cfg = getattr(cfg_exp, "distillation", None)
        self._distill_enabled = distill_cfg is not None and getattr(distill_cfg, "enabled", False)
        self._distill_weight = float(getattr(distill_cfg, "weight", 1.0)) if self._distill_enabled else 0.0

        # Delta preservation: cosine + norm penalties on cross-attn deltas vs parent.
        delta_pres_cfg = getattr(cfg_exp, "delta_preservation", None)
        self._delta_pres_enabled = delta_pres_cfg is not None and getattr(delta_pres_cfg, "enabled", False)
        self._delta_pres_weight_dir = float(getattr(delta_pres_cfg, "weight_dir", 0.1)) if self._delta_pres_enabled else 0.0
        self._delta_pres_weight_norm = float(getattr(delta_pres_cfg, "weight_norm", 0.01)) if self._delta_pres_enabled else 0.0

        # CoM separation loss: hinge on binder-target center-of-mass distance.
        # Prevents binder-target overlap during alignment-only training (dlw=0).
        self._com_separation_weight = float(
            getattr(loss_cfg, "com_separation_weight", 0.0)
        )

        # Gate magnitude penalty: hinge loss on |tanh(gate)| to push gates
        # toward small values during alignment training.
        gate_penalty_cfg = getattr(cfg_exp, "gate_penalty", None)
        self._gate_penalty_weight = float(getattr(gate_penalty_cfg, "weight", 0.0)) if gate_penalty_cfg is not None else 0.0
        self._gate_penalty_threshold = float(getattr(gate_penalty_cfg, "threshold", 0.01)) if gate_penalty_cfg is not None else 0.01

        # Learnable grounding layer weights (optional)
        self.grounding_layer_weights = None
        aux_losses = getattr(loss_cfg, "aux_losses", None) if loss_cfg is not None else None
        if aux_losses is not None:
            grounding_cfg = getattr(aux_losses, "grounding_loss", None)
            if grounding_cfg is not None:
                use_learned = getattr(grounding_cfg, "learned_layer_weights", False)
                if use_learned and hasattr(self.nn, "text_cross_attn_layers") and self.nn.text_cross_attn_layers is not None:
                    n_cross_attn = len(self.nn.text_cross_attn_layers)
                    from proteinfoundation.nn.modules.grounding import GroundingLayerWeights
                    self.grounding_layer_weights = GroundingLayerWeights(n_cross_attn)
                    logger.info(f"Grounding: learnable layer weights for {n_cross_attn} layers")

        # Alignment head (optional)
        self.alignment_head = None
        self._align_enabled = False
        self._align_extended_grounding = False
        self._align_binder_loss_weight = 1.0
        self._align_target_loss_weight = 1.0
        align_cfg = getattr(cfg_exp, "alignment_head", None)
        if align_cfg is not None and getattr(align_cfg, "enabled", False):
            from proteinfoundation.nn.modules.alignment_head import AlignmentHead

            source_type = align_cfg.source_type
            self._align_loss_weight = float(getattr(align_cfg, "loss_weight", 0.1))
            self._align_extended_grounding = bool(getattr(align_cfg, "extended_grounding", False))
            self._align_binder_loss_weight = float(getattr(align_cfg, "binder_loss_weight", 1.0))
            self._align_target_loss_weight = float(getattr(align_cfg, "target_loss_weight", 1.0))
            if self._align_extended_grounding and source_type != "cross_attn_delta":
                raise ValueError(
                    "alignment_head.extended_grounding=true currently requires "
                    "source_type='cross_attn_delta'"
                )

            # Determine input dimension based on source type
            if source_type in ("block_hidden", "block_hidden_last", "cross_attn_delta"):
                d_model = cfg_exp.nn.token_dim
            elif source_type == "output_bb_ca":
                d_model = 3
            elif source_type == "output_local_latents":
                if not hasattr(self, "latent_dim"):
                    raise ValueError(
                        "source_type='output_local_latents' requires local_latents flowmatcher"
                    )
                d_model = self.latent_dim
            elif source_type == "output_both":
                if not hasattr(self, "latent_dim"):
                    raise ValueError(
                        "source_type='output_both' requires local_latents flowmatcher"
                    )
                d_model = 3 + self.latent_dim
            else:
                raise ValueError(f"Unknown alignment source_type: {source_type}")

            # n_layers for multiscale sources
            n_layers = None
            if source_type == "block_hidden":
                n_layers = cfg_exp.nn.nlayers
            elif source_type == "cross_attn_delta":
                if not hasattr(self.nn, "text_cross_attn_layers") or self.nn.text_cross_attn_layers is None:
                    raise ValueError(
                        "source_type='cross_attn_delta' requires use_text_cross_attn=True"
                    )
                n_layers = len(self.nn.text_cross_attn_layers)

            # Text tokens are projected to token_dim by the tokens projector
            d_text = cfg_exp.nn.token_dim

            timestep_power = float(getattr(align_cfg, "timestep_power", 0.0))
            self.alignment_head = AlignmentHead(
                d_model=d_model,
                d_text=d_text,
                source_type=source_type,
                n_layers=n_layers,
                timestep_power=timestep_power,
            )
            self._align_enabled = True
            logger.info(
                f"AlignmentHead: source={source_type}, d_model={d_model}, "
                f"d_text={d_text}, loss_weight={self._align_loss_weight}, "
                f"timestep_power={timestep_power}, "
                f"extended_grounding={self._align_extended_grounding}, "
                f"binder_loss_weight={self._align_binder_loss_weight}, "
                f"target_loss_weight={self._align_target_loss_weight}"
            )


        # Apply freezing (must come after all modules are initialized)
        self._apply_freezing()

        # Distillation frozen state — populated in on_fit_start (after
        # checkpoint loading) so the snapshot contains parent weights.
        self._distill_frozen_state = None

        # Scaling laws stuff
        self.nflops = 0
        self.nsamples_processed = 0
        self.nparams = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # For autoguidance, overridden in `self.configure_inference`
        self.nn_ag = None

    def setup(self, stage: str | None = None) -> None:
        """Move frozen text encoder models to GPU after DDP setup.

        Called by PL after DDP process group init but before model_to_device().
        self.device is pre-patched to root_device by PL before this hook.
        The barrier("post_setup") in _call_setup_hook ensures all DDP
        ranks perform this slow GPU transfer together, preventing the
        startup-time rank desync that causes DDP all-reduce hangs on
        the first training_step when resuming with ckpt_path.
        """
        if stage != "fit" or self.text_encoder is None:
            return
        device = self.device
        if self.text_encoder._pooled_model is not None:
            self.text_encoder._pooled_model.to(device)
        if (
            self.text_encoder._tokens_model is not None
            and self.text_encoder._tokens_model is not self.text_encoder._pooled_model
        ):
            self.text_encoder._tokens_model.to(device)

    def on_fit_start(self):
        """Snapshot trainable params for distillation after checkpoint loading.

        Called by Lightning after checkpoint restore but before the first
        training step, so the snapshot contains the pretrained parent weights.
        """
        if not self._distill_enabled and not self._delta_pres_enabled:
            return
        self._distill_frozen_state = {
            name: param.data.clone()
            for name, param in self.nn.named_parameters()
            if param.requires_grad
        }
        logger.info(
            f"Distillation: snapshotted {len(self._distill_frozen_state)} "
            f"trainable param tensors for parent output recovery"
        )

    def load_autoencoder(self, cfg_exp, freeze_params=True):
        """Loads autoencoder, if required."""
        if "autoencoder_ckpt_path" in cfg_exp:  # for new runs trained with refactored codebase
            ae_ckp_path = cfg_exp.autoencoder_ckpt_path
        elif (
            "autoencoder_ckpt_path" in cfg_exp.product_flowmatcher.local_latents
        ):  # for old runs trained with old codebase
            ae_ckp_path = cfg_exp.product_flowmatcher.local_latents.autoencoder_ckpt_path
        else:
            raise ValueError("No autoencoder checkpoint path provided")

        if ae_ckp_path is None:
            return None, None

        # Load and freeze parameters
        autoencoder = AutoEncoder.load_from_checkpoint(ae_ckp_path)
        if freeze_params:
            for param in autoencoder.parameters():
                param.requires_grad = False
        return autoencoder, autoencoder.latent_dim

    @property
    def strict_loading(self):
        """Non-strict when text conditioning, alignment head, or adapter is enabled
        (allows loading base checkpoints that lack these module weights)."""
        has_adapter = hasattr(self.nn, "use_text_cross_attn_adapter") and self.nn.use_text_cross_attn_adapter
        return not (self._text_cond_enabled or self._align_enabled or has_adapter)

    def _apply_freezing(self):
        """Centralized freezing logic based on training.freeze config.
        Called after all modules are initialized."""
        training_cfg = getattr(self.cfg_exp, "training", None)
        if training_cfg is None:
            return

        freeze_cfg = getattr(training_cfg, "freeze", None)
        if freeze_cfg is None:
            return

        # 1. Freeze entire denoiser if requested
        freeze_denoiser = getattr(freeze_cfg, "denoiser", False)
        if freeze_denoiser:
            for p in self.nn.parameters():
                p.requires_grad = False

        # 2. Selectively unfreeze AdaLN layers and conditioning projection
        unfreeze_adaln = not getattr(freeze_cfg, "adaln_layers", True)
        if unfreeze_adaln and freeze_denoiser:
            from proteinfoundation.nn.modules.adaptive_ln_scale import (
                AdaptiveLayerNorm,
                AdaptiveOutputScale,
            )
            for module in self.nn.modules():
                if isinstance(module, (AdaptiveLayerNorm, AdaptiveOutputScale)):
                    for p in module.parameters():
                        p.requires_grad = True
            # Unfreeze cond_factory.linear_out — this projects feats_cond_seq
            # (including text_cond) into the conditioning signal for AdaLN.
            # Must be trainable when AdaLN is trainable, especially after
            # reinitialization due to text_cond adding a new input dimension.
            for factory_name in ("cond_factory", "pair_repr_builder"):
                factory = getattr(self.nn, factory_name, None)
                if factory is None:
                    continue
                # Handle nested pair_repr_builder.cond_factory
                if hasattr(factory, "cond_factory"):
                    factory = factory.cond_factory
                if hasattr(factory, "linear_out"):
                    for p in factory.linear_out.parameters():
                        p.requires_grad = True

            # 2b. Optionally unfreeze the conditioning FEATURE modules themselves.
            # Text's conditioning signal is produced by self.text_encoder, which
            # lives OUTSIDE self.nn and so escapes the denoiser freeze in step 1.
            # A feature that produces its signal INSIDE nn -- e.g.
            # FoldEmbeddingSeqFeat's CATH embedding tables -- is caught by that
            # freeze, so an AdaLN warm-up would train against a randomly
            # initialized signal. Set freeze.cond_features=false for parity.
            # Defaults to True (frozen), i.e. every existing recipe is unchanged.
            unfreeze_cond_feats = not getattr(freeze_cfg, "cond_features", True)
            if unfreeze_cond_feats:
                cond_factory = getattr(self.nn, "cond_factory", None)
                if cond_factory is not None:
                    n_unfrozen = 0
                    for p in cond_factory.parameters():
                        p.requires_grad = True
                        n_unfrozen += 1
                    logger.info(
                        f"[freeze] cond_features=false: unfroze {n_unfrozen} parameter "
                        f"tensors in nn.cond_factory (conditioning feature modules)"
                    )

        # 3. Unfreeze cross-attention layers if requested
        unfreeze_cross_attn = not getattr(freeze_cfg, "cross_attn", True)
        freeze_qk = getattr(freeze_cfg, "cross_attn_qk", False)
        freeze_gates_cfg = getattr(freeze_cfg, "gates", None)
        if unfreeze_cross_attn and freeze_denoiser:
            if hasattr(self.nn, "text_cross_attn_layers") and self.nn.text_cross_attn_layers is not None:
                for p in self.nn.text_cross_attn_layers.parameters():
                    p.requires_grad = True

            def _freeze_cross_attn_conditioning(layer):
                for p in layer.adaln.parameters():
                    p.requires_grad = False
                for p in layer.ln_kv.parameters():
                    p.requires_grad = False
                if hasattr(layer, "scale_output"):
                    for p in layer.scale_output.parameters():
                        p.requires_grad = False
                if hasattr(layer, "output_gate"):
                    layer.output_gate.requires_grad = False

            # 3a. Fine-grained Q/K freezing via gradient hooks.
            # Packed in_proj_weight is [3*E, E]: rows [:E]=Q, [E:2E]=K, [2E:]=V.
            # Hooks zero Q/K gradient slices so only V weights update.
            if freeze_qk and hasattr(self.nn, "text_cross_attn_layers") and self.nn.text_cross_attn_layers is not None:
                def _make_qk_zero_hook(embed_dim):
                    def hook(grad):
                        grad[:2 * embed_dim].zero_()
                        return grad
                    return hook

                for layer_key, layer in self.nn.text_cross_attn_layers.items():
                    # Re-freeze conditioning/output-scaling path around attention.
                    _freeze_cross_attn_conditioning(layer)
                    E = layer.attn.embed_dim
                    if not getattr(layer.attn.in_proj_weight, '_qk_hook_registered', False):
                        layer.attn.in_proj_weight.register_hook(_make_qk_zero_hook(E))
                        layer.attn.in_proj_weight._qk_hook_registered = True
                        if layer.attn.in_proj_bias is not None:
                            layer.attn.in_proj_bias.register_hook(_make_qk_zero_hook(E))
                    logger.info(
                        f"text_cross_attn layer {layer_key}: Q=frozen K=frozen V=train out=train"
                    )

            # Re-freeze cross-attn adaln/ln_kv without Q/K hooks (for QKV-trainable controls)
            freeze_xattn_adaln = getattr(freeze_cfg, "cross_attn_adaln", False)
            if freeze_xattn_adaln and not freeze_qk and hasattr(self.nn, "text_cross_attn_layers") and self.nn.text_cross_attn_layers is not None:
                for layer_key, layer in self.nn.text_cross_attn_layers.items():
                    _freeze_cross_attn_conditioning(layer)
                    logger.info(f"text_cross_attn layer {layer_key}: conditioning/output-scale frozen, Q/K/V/out train")

        # 3b. Layer-restricted cross-attn: re-freeze layers not in subset
        cross_attn_layers_cfg = getattr(freeze_cfg, "cross_attn_layers", None)
        if cross_attn_layers_cfg is not None and unfreeze_cross_attn and freeze_denoiser:
            if hasattr(self.nn, "text_cross_attn_layers") and self.nn.text_cross_attn_layers is not None:
                allowed = set(str(i) for i in cross_attn_layers_cfg)
                for layer_key, layer in self.nn.text_cross_attn_layers.items():
                    if layer_key not in allowed:
                        for p in layer.parameters():
                            p.requires_grad = False
                        logger.info(f"text_cross_attn layer {layer_key}: re-frozen (not in layer subset)")

        # 3c. Last-block unfreeze (for block_hidden_last alignment)
        # Only AdaLN + transition, not self-attention Q/K/V (per codex plan).
        last_block_unfreeze = getattr(freeze_cfg, "last_block_unfreeze", False)
        if last_block_unfreeze and freeze_denoiser:
            if hasattr(self.nn, "transformer_layers"):
                from proteinfoundation.nn.modules.adaptive_ln_scale import (
                    AdaptiveLayerNorm,
                    AdaptiveOutputScale,
                )
                last_block = self.nn.transformer_layers[-1]
                n_unfrozen = 0
                for module in last_block.modules():
                    if isinstance(module, (AdaptiveLayerNorm, AdaptiveOutputScale)):
                        for p in module.parameters():
                            p.requires_grad = True
                            n_unfrozen += p.numel()
                if hasattr(last_block, "transition"):
                    for p in last_block.transition.parameters():
                        p.requires_grad = True
                        n_unfrozen += p.numel()
                logger.info(f"Last transformer block: AdaLN+transition unfrozen ({n_unfrozen:,} params)")

        # 3d. Gates: independently controllable (trainable even when cross_attn frozen)
        if freeze_denoiser and hasattr(self.nn, "text_cross_attn_gates") and self.nn.text_cross_attn_gates is not None:
            if freeze_gates_cfg is None:
                gates_trainable = unfreeze_cross_attn
            else:
                gates_trainable = not freeze_gates_cfg
            for p in self.nn.text_cross_attn_gates.parameters():
                p.requires_grad = gates_trainable

        # 4. Handle text projectors (default: trainable unless explicitly frozen)
        # Projectors live on self.text_encoder (not self.nn), so step 1's
        # denoiser freeze does not reach them. Must explicitly set requires_grad.
        freeze_projectors = getattr(freeze_cfg, "projectors", False)
        if self.text_encoder is not None:
            if self.text_encoder.pooled_proj is not None:
                for p in self.text_encoder.pooled_proj.parameters():
                    p.requires_grad = not freeze_projectors
            if self.text_encoder.tokens_proj is not None:
                for p in self.text_encoder.tokens_proj.parameters():
                    p.requires_grad = not freeze_projectors

        # 5. Adapter cross-attention layers (default: trainable when they exist).
        # Step 1 freezes the entire denoiser including adapter layers; unfreeze here.
        freeze_adapter = getattr(freeze_cfg, "adapter", False)
        if not freeze_adapter and freeze_denoiser:
            if hasattr(self.nn, "text_cross_attn_adapter_layers") and self.nn.text_cross_attn_adapter_layers is not None:
                for p in self.nn.text_cross_attn_adapter_layers.parameters():
                    p.requires_grad = True
            if hasattr(self.nn, "text_cross_attn_adapter_gates") and self.nn.text_cross_attn_adapter_gates is not None:
                for p in self.nn.text_cross_attn_adapter_gates.parameters():
                    p.requires_grad = True

        # 6. Alignment head (default: trainable unless explicitly frozen)
        freeze_alignment_head = getattr(freeze_cfg, "alignment_head", False)
        if self.alignment_head is not None:
            for p in self.alignment_head.parameters():
                p.requires_grad = not freeze_alignment_head

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Freezing applied: {n_trainable:,} / {n_total:,} parameters trainable"
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.cfg_exp.opt.lr)
        warmup_steps = self.cfg_exp.opt.get("warmup_steps", 0)
        if warmup_steps > 0:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint.
        Excludes frozen text encoder weights (safety net — they should
        already be invisible via object.__setattr__, but belt-and-suspenders)."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

        if self.text_encoder is not None:
            keys_to_remove = [
                k for k in checkpoint["state_dict"]
                if "text_encoder._pooled_model" in k
                or "text_encoder._tokens_model" in k
            ]
            for k in keys_to_remove:
                del checkpoint["state_dict"][k]

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except (KeyError, AttributeError):
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

    def call_nn(
        self,
        batch: dict[str, torch.Tensor],
        n_recycle: int = 0,
    ) -> dict[str, torch.Tensor]:
        """
        Calls NN with recycling. Should this be here or in the NN? Possibly better here,
        in case we want to recycle using decoder for some approach, etc, and this is akin
        to self conditioning, also here.
        Also, if we want to recycle clean sample predictions... Then we'd need this here,
        as the nn does not know about relations between v, x1, ...
        """
        # First call
        nn_out = self.nn(batch)

        # Recycle n_recycle times detaching gradients and updating input
        for _ in range(n_recycle):
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)
            batch["x_recycle"] = {dm: x_1_pred[dm].detach() for dm in x_1_pred}
            nn_out = self.nn(batch)

        # Final prediction
        return nn_out

    def _assemble_align_intermediates(self, nn_out: dict) -> dict:
        """Build the intermediates dict expected by AlignmentHead._get_source."""
        intermediates = {}
        source = self.alignment_head.source_type

        if source == "block_hidden":
            intermediates["block_hidden"] = nn_out["block_hidden"]
        elif source == "block_hidden_last":
            intermediates["block_hidden_last"] = nn_out["block_hidden_last"]
        elif source == "cross_attn_delta":
            key = "cross_attn_deltas_extended" if self._align_extended_grounding else "cross_attn_deltas"
            intermediates["cross_attn_deltas"] = nn_out[key]
        elif source in ("output_bb_ca", "output_both"):
            # nn_out["bb_ca"] is {param_key: tensor} — extract the tensor
            intermediates["bb_ca"] = next(iter(nn_out["bb_ca"].values()))
        if source in ("output_local_latents", "output_both"):
            intermediates["local_latents"] = next(iter(nn_out["local_latents"].values()))

        return intermediates

    def _ensure_compact_concat_layout(self, batch: dict) -> bool:
        """Populate compact binder/target index maps when target concat exists."""
        if "binder_idx_ext" in batch and "target_idx_ext" in batch:
            return True
        if "mask" not in batch or "seq_target_mask" not in batch:
            return False
        from proteinfoundation.utils.tensor_utils import compact_concat_indices

        layout = compact_concat_indices(
            batch["mask"].bool(),
            batch["seq_target_mask"].bool(),
        )
        batch.update(layout)
        return True

    def _apply_fact_dropout(self, batch, cfg_keep_mask):
        """Zero token embeddings for a randomly selected fact per sample.

        Uses alignment annotations to identify which token positions correspond
        to a specific fact (e.g., "3 disulfide bonds"). Zeroing those tokens
        teaches the model which text facts drive which structural features,
        making the CFG mechanism fact-specific at inference time.

        Only modifies token_cond (cross-attention path), not pooled_cond (AdaLN).
        This creates a useful asymmetry: AdaLN still sees the full description
        while cross-attention misses one fact, teaching cross-attention to carry
        fact-specific information.
        """
        import random as _random
        from proteinfoundation.nn.modules.grounding import _tokenize_with_offsets

        token_cond = batch["text_tokens"]  # [B, L, token_dim]
        texts = batch["text"]
        alignments_list = batch["text_alignments"]
        tokenizer = self.text_encoder._tokens_tokenizer
        use_chat = self.text_encoder._tokens_use_chat_template
        b = token_cond.shape[0]

        # Batch-tokenize once to get offset mappings (reuse grounding.py logic)
        offset_mapping, char_offsets = _tokenize_with_offsets(
            texts, tokenizer, use_chat,
        )  # offset_mapping: [B, L_tok, 2], char_offsets: [B]

        # Build a per-token drop mask [B, L, 1] (1.0 = keep, 0.0 = drop)
        L_tok = offset_mapping.shape[1]
        L_cond = token_cond.shape[1]
        L = min(L_tok, L_cond)
        drop_mask = torch.ones(b, L_cond, 1, device=token_cond.device, dtype=token_cond.dtype)

        for i in range(b):
            # Skip CFG-dropped samples (already zeroed)
            if cfg_keep_mask is not None and not cfg_keep_mask[i]:
                continue
            if _random.random() >= self._p_fact_drop:
                continue

            aligns = alignments_list[i]
            if not aligns:
                continue

            # Filter to implicit annotations with text_chars
            valid = [a for a in aligns if a.get("type") != "fallback" and a.get("text_chars")]
            if not valid:
                continue

            chosen = _random.choice(valid)
            char_start, char_end = chosen["text_chars"]
            adj_start = char_start + char_offsets[i]
            adj_end = char_end + char_offsets[i]

            offsets = offset_mapping[i]  # [L_tok, 2]
            for t_idx in range(L):
                t_start, t_end = offsets[t_idx].tolist()
                # Skip special tokens (BOS/EOS/PAD have offset (0,0))
                if t_start == 0 and t_end == 0:
                    continue
                if t_end > adj_start and t_start < adj_end:
                    drop_mask[i, t_idx, 0] = 0.0

        # Out-of-place multiply preserves autograd graph
        batch["text_tokens"] = token_cond * drop_mask

    def predict_for_sampling(
        self,
        batch: dict,
        mode: Literal["full", "ucond", "contrastive"] = "full",
        n_recycle: int = 0,
    ) -> tuple[dict[str, torch.Tensor] | float | None]:
        """
        This function predicts clean samples for multiple models:
        x_pred, the 'original' model, if mode == full
        x_pred_ucond, the unconditional model, if mode == ucond
        x_pred_contrastive, the contrastive baseline model (y⁻), if mode == contrastive

        These predictions will later be used to sample with guidance and autoguidance.

        Args:
            batch: Dict
            mode: str

        Returns:
            x_pred (tensor) for the requested mode
        """
        if mode == "full":
            nn_out = self.call_nn(batch, n_recycle=n_recycle)
        elif mode == "ucond":
            uncond_batch = batch.copy()
            # Zero text conditioning tensors (not pop) to match training
            # CFG dropout behavior. LN(zeros) produces non-zero output from
            # bias terms, so the unconditional signal must be consistent
            # between training (zeroed embeddings, cross-attn runs) and
            # inference (also zeroed embeddings, cross-attn runs).
            # text_tokens_mask is intentionally left unchanged — zeroed values
            # with valid mask produce zero cross-attn output, matching training.
            if "text_cond" in uncond_batch:
                uncond_batch["text_cond"] = torch.zeros_like(uncond_batch["text_cond"])
            if "text_tokens" in uncond_batch:
                uncond_batch["text_tokens"] = torch.zeros_like(uncond_batch["text_tokens"])
            # Pop fold conditioning
            uncond_batch.pop("cath_code", None)
            nn_out = self.call_nn(uncond_batch, n_recycle=n_recycle)
        elif mode == "contrastive":
            # Swap in baseline text embeddings (y⁻) for contrastive guidance.
            contra_batch = batch.copy()
            if "text_cond_baseline" in contra_batch:
                contra_batch["text_cond"] = contra_batch.pop("text_cond_baseline")
            if "text_tokens_baseline" in contra_batch:
                contra_batch["text_tokens"] = contra_batch.pop("text_tokens_baseline")
                contra_batch["text_tokens_mask"] = contra_batch.pop("text_tokens_mask_baseline")
            nn_out = self.call_nn(contra_batch, n_recycle=n_recycle)
        else:
            raise OSError(f"Wrong {mode} passed to `predict_for_sampling`")

        return nn_out

    def skip_forward_pass(self, batch: dict, batch_idx: int):
        """
        Skips the forward pass and returns 0.
        """
        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def on_train_epoch_start(self):
        if self.trainer is not None and self.trainer.max_epochs is not None:
            self.fm.set_total_epochs(self.trainer.max_epochs)
        self.fm.set_epoch(self.current_epoch)
        # Propagate total_epochs to ClusterSampler for β annealing
        dm = getattr(self.trainer, "datamodule", None)
        sampler = getattr(dm, "_train_sampler", None) if dm is not None else None
        if sampler is not None and hasattr(sampler, "set_total_epochs"):
            if self.trainer.max_epochs is not None:
                # If extending training, unlock and apply new schedule params
                extend = self.cfg_exp.get("extend_beta_schedule", None)
                if extend is not None and hasattr(sampler, "extend_schedule"):
                    sampler.extend_schedule(
                        total_epochs=extend.get("total_epochs", self.trainer.max_epochs),
                        beta_n_cycles=extend.get("beta_n_cycles", None),
                    )
                else:
                    sampler.set_total_epochs(self.trainer.max_epochs)

    def training_step(self, batch: dict, batch_idx: int):
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batch dimension.
        """
        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"

        batch = add_clean_samples(
            batch,
            self.cfg_exp.product_flowmatcher,
            getattr(self, "autoencoder", None),
        )

        # For editing pairs: swap x_1 to target encoding, save input as x_1_input
        if "edit_coords_nm" in batch:
            from proteinfoundation.utils.sample_utils import encode_edit_target
            batch["x_1_input"] = {k: v.clone() for k, v in batch["x_1"].items()}
            batch["x_1"] = encode_edit_target(
                batch, self.cfg_exp.product_flowmatcher,
                getattr(self, "autoencoder", None),
            )
            # Build per-residue edit site mask for diff-weighted loss
            batch["edit_site_mask"] = (
                batch["edit_residue_type"] != batch["residue_type"]
            )  # [B, N] bool
            # Free AE intermediate memory before denoiser forward pass
            torch.cuda.empty_cache()

        # Corrupt the batch
        batch = self.fm.corrupt_batch(batch)  # adds x_1, t, x_0, x_t, mask
        bs, n = batch["mask"].shape

        # Text conditioning (before self-cond so self-cond sees text,
        # matching la-proteina's ordering).
        # NOTE: every batch MUST have "text" when text conditioning + freeze
        # is active, otherwise cross-attn layer parameters are unused in the
        # forward pass and DDP (find_unused_parameters=False) will crash.
        # Ensure the data pipeline always provides text via AddTextFromCSV.
        if self.text_encoder is not None and "text" in batch:
            orig_p_uncond = self.text_encoder.p_uncond
            t_dict = batch.get("t", {})
            t_for_dropout = t_dict.get("bb_ca")
            if t_for_dropout is None and t_dict:
                t_for_dropout = next(iter(t_dict.values()))
            pooled_cond, token_cond, token_mask, cfg_keep_mask = self.text_encoder(
                batch["text"], device=batch["mask"].device, training=self.training,
                t=t_for_dropout,
            )
            self.text_encoder.p_uncond = orig_p_uncond
            if pooled_cond is not None:
                batch["text_cond"] = pooled_cond
            if token_cond is not None:
                batch["text_tokens"] = token_cond
                batch["text_tokens_mask"] = token_mask

            # Fact-level dropout: zero token embeddings for a random fact's
            # text span, using alignment annotations to identify spans.
            # Only applies to samples that weren't CFG-dropped.
            if (
                self._p_fact_drop > 0.0
                and self.training
                and token_cond is not None
                and "text_alignments" in batch
                and "text" in batch
            ):
                self._apply_fact_dropout(batch, cfg_keep_mask)

            # Build grounding masks for cross-attention supervision.
            # Only when grounding_loss is configured — otherwise skip to
            # avoid the perf hit of need_weights=True (disables flash attn).
            has_cross_attn_grounding_loss = (
                hasattr(self.cfg_exp, "loss")
                and "grounding_loss" in getattr(self.cfg_exp.loss, "aux_losses", {})
            )
            has_grounding_loss = has_cross_attn_grounding_loss or self._align_enabled
            # Only request attention maps when the cross-attn grounding loss
            # needs them — the alignment head does NOT need attention maps.
            if has_cross_attn_grounding_loss:
                batch["collect_text_attn_maps"] = True
            if (
                has_grounding_loss
                and token_cond is not None
                and "text_alignments" in batch
                and "text_modelled_positions" in batch
            ):
                if self._align_extended_grounding and self._ensure_compact_concat_layout(batch):
                    from proteinfoundation.nn.modules.grounding import (
                        build_extended_grounding_masks,
                        target_positions_from_batch,
                    )

                    target_positions = target_positions_from_batch(batch)
                    grounding_mask, token_has_mask, grounding_stats = build_extended_grounding_masks(
                        texts=batch["text"],
                        alignments_list=batch["text_alignments"],
                        binder_modelled_positions_list=batch["text_modelled_positions"],
                        target_modelled_positions_list=target_positions,
                        tokenizer=self.text_encoder._tokens_tokenizer,
                        binder_idx_ext=batch["binder_idx_ext"],
                        target_idx_ext=batch["target_idx_ext"],
                        device=batch["mask"].device,
                        use_chat_template=self.text_encoder._tokens_use_chat_template,
                    )
                    batch["grounding_protein_mask"] = (
                        batch["binder_mask_ext"] | batch["target_mask_ext"]
                    )
                    if (
                        self._align_binder_loss_weight != 1.0
                        or self._align_target_loss_weight != 1.0
                    ):
                        batch["grounding_residue_weight"] = torch.where(
                            batch["target_mask_ext"],
                            torch.full_like(
                                batch["target_mask_ext"],
                                self._align_target_loss_weight,
                                dtype=torch.float32,
                            ),
                            torch.full_like(
                                batch["binder_mask_ext"],
                                self._align_binder_loss_weight,
                                dtype=torch.float32,
                            ),
                        )
                    if grounding_stats["n_residues_total"] > 0:
                        logger.info(
                            "Extended grounding: residues %d/%d resolved",
                            grounding_stats["n_residues_resolved"],
                            grounding_stats["n_residues_total"],
                        )
                else:
                    from proteinfoundation.nn.modules.grounding import build_grounding_masks

                    # Use residue_pdb_idx shape as authoritative original residue
                    # count — it is never extended by concat features.
                    if "residue_pdb_idx" in batch:
                        n_res_grounding = batch["residue_pdb_idx"].shape[1]
                    else:
                        n_res_grounding = batch["mask"].shape[1]

                    grounding_mask, token_has_mask, grounding_stats = build_grounding_masks(
                        texts=batch["text"],
                        alignments_list=batch["text_alignments"],
                        modelled_positions_list=batch["text_modelled_positions"],
                        tokenizer=self.text_encoder._tokens_tokenizer,
                        n_residues=n_res_grounding,
                        device=batch["mask"].device,
                        use_chat_template=self.text_encoder._tokens_use_chat_template,
                        res_mask=batch["mask"][:, :n_res_grounding],
                    )
                    batch["grounding_protein_mask"] = batch["mask"][:, :n_res_grounding]
                    if grounding_stats["n_skipped_mismatch"] > 0:
                        logger.info(
                            "Grounding: %d/%d samples skipped (res_mask mismatch)",
                            grounding_stats["n_skipped_mismatch"],
                            grounding_stats["n_samples"],
                        )
                # Zero out grounding for CFG-dropped samples
                if cfg_keep_mask is not None:
                    token_has_mask = token_has_mask & cfg_keep_mask[:, None]
                batch["grounding_mask"] = grounding_mask
                batch["token_has_mask"] = token_has_mask
                if self.grounding_layer_weights is not None:
                    batch["grounding_layer_weights"] = self.grounding_layer_weights

            if cfg_keep_mask is not None:
                batch["cfg_keep_mask"] = cfg_keep_mask

        # Handle conditioning variables (safe config getters; missing keys default to disabled)
        batch, n_recycle = handle_batch_conditioning(
            batch,
            bs,
            self.cfg_exp.training,
            self.call_nn,
            self.fm,
        )

        # Set intermediate collection flags for alignment head inputs (skip on val)
        if self._align_enabled and not val_step:
            if self._align_enabled and self.alignment_head.needs_block_hidden:
                batch["collect_block_hidden"] = True
            elif self._align_enabled and self.alignment_head.needs_block_hidden_last:
                batch["collect_block_hidden_last"] = True
            elif self._align_enabled and self.alignment_head.needs_cross_attn_deltas:
                if self._align_extended_grounding:
                    batch["collect_cross_attn_deltas_extended"] = True
                else:
                    batch["collect_cross_attn_deltas"] = True

        nn_out = self.call_nn(batch, n_recycle=n_recycle)

        losses = self.fm.compute_loss(
            batch=batch,
            nn_out=nn_out,
        )  # Dict[str, Tensor w.batch shape [*]]

        # Alignment head loss
        if (
            self._align_enabled
            and not val_step
            and "grounding_mask" in batch
            and "token_has_mask" in batch
            and "text_tokens" in batch
        ):
            intermediates = self._assemble_align_intermediates(nn_out)
            # Use bb_ca timestep for weighting (shared across modes in practice)
            t = batch["t"].get("bb_ca", next(iter(batch["t"].values())))
            align_loss = self.alignment_head(
                intermediates=intermediates,
                text_tokens=batch["text_tokens"],
                grounding_mask=batch["grounding_mask"],
                token_has_mask=batch["token_has_mask"],
                protein_mask=batch.get(
                    "grounding_protein_mask",
                    batch["mask"][:, :batch["grounding_mask"].shape[2]],
                ),
                residue_weight=batch.get("grounding_residue_weight", None),
                t=t,
            )  # [B]
            losses["alignment_head"] = align_loss * self._align_loss_weight
            losses["alignment_head_raw_justlog"] = align_loss.detach()
        elif self._align_enabled and not val_step:
            # DDP guard: alignment head params must participate in backward.
            # Must NOT be _justlog — needs to be in train_loss for .backward()
            # to touch these params, keeping DDP (find_unused_parameters=False) happy.
            dummy = self.alignment_head.protein_proj.weight.sum() * 0.0
            if hasattr(self.alignment_head, "layer_logits"):
                dummy = dummy + self.alignment_head.layer_logits.sum() * 0.0
            # With detach_cross_attn_delta, FM loss cannot reach cross-attn params.
            # Anchor them here so DDP (find_unused_parameters=False) does not crash.
            if getattr(self.nn, "detach_cross_attn_delta", False):
                if hasattr(self.nn, "text_cross_attn_layers") and self.nn.text_cross_attn_layers is not None:
                    for layer in self.nn.text_cross_attn_layers.values():
                        dummy = dummy + sum(p.sum() for p in layer.parameters()) * 0.0
                if hasattr(self.nn, "text_cross_attn_gates") and self.nn.text_cross_attn_gates is not None:
                    for p in self.nn.text_cross_attn_gates.parameters():
                        dummy = dummy + p.sum() * 0.0
            losses["alignment_head"] = dummy

        # DDP anchor for adapter params — independent of alignment head.
        # Only fires when trainable adapter params exist and no grounding data
        # is present (the alignment loss path already touches adapter params
        # when grounding data exists).
        if (
            not val_step
            and hasattr(self.nn, "text_cross_attn_adapter_layers")
            and self.nn.text_cross_attn_adapter_layers is not None
            and "grounding_mask" not in batch
        ):
            trainable_adapter_params = [
                p for layer in self.nn.text_cross_attn_adapter_layers.values()
                for p in layer.parameters() if p.requires_grad
            ]
            if hasattr(self.nn, "text_cross_attn_adapter_gates") and self.nn.text_cross_attn_adapter_gates is not None:
                trainable_adapter_params += [
                    p for p in self.nn.text_cross_attn_adapter_gates.parameters()
                    if p.requires_grad
                ]
            if trainable_adapter_params:
                losses["adapter_ddp_anchor"] = sum(
                    p.sum() * 0.0 for p in trainable_adapter_params
                )

        # Output distillation to parent: MSE between model and parent outputs.
        # Recovers parent output by swapping in frozen (init) weights for all
        # trainable params, forwarding, then restoring current weights.
        if self._distill_enabled and not val_step:
            with torch.no_grad():
                batch_parent = dict(batch)
                for _flag in (
                    "collect_block_hidden", "collect_block_hidden_last",
                    "collect_cross_attn_deltas", "collect_cross_attn_deltas_extended",
                    "collect_pair_rep",
                ):
                    batch_parent.pop(_flag, None)
                # Swap in frozen parent weights.
                saved = {}
                nn_params = dict(self.nn.named_parameters())
                for name, frozen_data in self._distill_frozen_state.items():
                    saved[name] = nn_params[name].data
                    if frozen_data.device != nn_params[name].data.device:
                        frozen_data = frozen_data.to(nn_params[name].data.device)
                        self._distill_frozen_state[name] = frozen_data
                    nn_params[name].data = frozen_data
                try:
                    nn_out_parent = self.call_nn(batch_parent, n_recycle=n_recycle)
                finally:
                    # Restore current weights even if forward raises.
                    for name, current_data in saved.items():
                        nn_params[name].data = current_data
            distill_loss = batch["mask"].new_zeros(())
            n_modes = 0
            for dm in self.fm.data_modes:
                v_model = nn_out[dm].get("v")
                v_parent = nn_out_parent[dm].get("v")
                if v_model is not None and v_parent is not None:
                    mask_dm = batch["mask"][..., None]  # [B, N, 1]
                    distill_loss = distill_loss + (
                        ((v_model - v_parent.detach()) ** 2 * mask_dm).sum()
                        / mask_dm.sum().clamp_min(1)
                    )
                    n_modes += 1
            if n_modes > 0:
                distill_loss = distill_loss / n_modes
            losses["distillation"] = distill_loss * self._distill_weight
            losses["distillation_raw_justlog"] = distill_loss.detach()

        # Delta preservation: cosine + norm penalties on cross-attn deltas vs parent.
        if self._delta_pres_enabled and not val_step:
            deltas_model = nn_out.get("cross_attn_deltas")
            if deltas_model is not None:
                with torch.no_grad():
                    batch_parent_dp = dict(batch)
                    batch_parent_dp["collect_cross_attn_deltas"] = True
                    saved_dp = {}
                    nn_params_dp = dict(self.nn.named_parameters())
                    for name, frozen_data in self._distill_frozen_state.items():
                        saved_dp[name] = nn_params_dp[name].data
                        if frozen_data.device != nn_params_dp[name].data.device:
                            frozen_data = frozen_data.to(nn_params_dp[name].data.device)
                            self._distill_frozen_state[name] = frozen_data
                        nn_params_dp[name].data = frozen_data
                    try:
                        nn_out_parent_dp = self.call_nn(batch_parent_dp, n_recycle=n_recycle)
                    finally:
                        for name, current_data in saved_dp.items():
                            nn_params_dp[name].data = current_data
                deltas_parent = nn_out_parent_dp.get("cross_attn_deltas")
                if deltas_parent is not None:
                    mask_dp = batch["mask"][:, None, :, None].float()
                    d_m = deltas_model * mask_dp
                    d_p = deltas_parent.detach() * mask_dp
                    cos_sim = torch.nn.functional.cosine_similarity(
                        d_m.flatten(2), d_p.flatten(2), dim=-1
                    )
                    dir_loss = (1.0 - cos_sim).mean()
                    norm_m = d_m.flatten(2).norm(dim=-1)
                    norm_p = d_p.flatten(2).norm(dim=-1).clamp(min=1e-8)
                    norm_loss = (norm_m / norm_p - 1.0).abs().mean()
                    losses["delta_pres_dir"] = dir_loss * self._delta_pres_weight_dir
                    losses["delta_pres_norm"] = norm_loss * self._delta_pres_weight_norm

        # CoM separation loss: hinge on binder-target center-of-mass distance.
        if (
            self._com_separation_weight > 0.0
            and not val_step
            and "x_target" in batch
            and "target_mask" in batch
        ):
            x1_pred_bb = nn_out["bb_ca"].get("x_1")
            if x1_pred_bb is None:
                # Compute x_1 from velocity: x_1 = x_t + (1-t) * v
                t_bb = batch["t"]["bb_ca"]
                x1_pred_bb = (
                    batch["x_t"]["bb_ca"]
                    + (1.0 - t_bb[..., None, None]) * nn_out["bb_ca"]["v"]
                ) * batch["mask"][..., None]
            binder_mask_f = batch["mask"].float()  # [B, N_orig]
            binder_n = binder_mask_f.sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            binder_com_pred = (x1_pred_bb * binder_mask_f[..., None]).sum(dim=1) / binder_n  # [B, 3]

            # target_mask is [B, N_target, 37] (per-atom); use CA (index 1) as residue mask
            target_mask_ca = batch["target_mask"][:, :, 1].float()  # [B, N_target]
            target_n = target_mask_ca.sum(dim=1, keepdim=True).clamp(min=1)
            # x_target is [B, N_target, 37, 3] (atom37); extract CA
            x_target_ca = batch["x_target"][:, :, 1, :]  # [B, N_target, 3]
            target_com = (x_target_ca * target_mask_ca[..., None]).sum(dim=1) / target_n  # [B, 3]

            binder_com_gt = (
                batch["x_1"]["bb_ca"] * binder_mask_f[..., None]
            ).sum(dim=1) / binder_n  # [B, 3]

            d_pred = (binder_com_pred - target_com).norm(dim=-1)  # [B]
            d_gt = (binder_com_gt - target_com).norm(dim=-1)  # [B]

            com_loss = torch.nn.functional.relu(d_gt - d_pred)  # [B]
            losses["com_separation"] = com_loss * self._com_separation_weight
            losses["com_separation_raw_justlog"] = com_loss.detach()
            losses["com_d_pred_justlog"] = d_pred.detach()
            losses["com_d_gt_justlog"] = d_gt.detach()


        if (
            self._gate_penalty_weight > 0
            and not val_step
            and hasattr(self.nn, "text_cross_attn_gates")
            and self.nn.text_cross_attn_gates is not None
        ):
            gate_hinge_list = [
                torch.relu(gate.tanh().abs() - self._gate_penalty_threshold)
                for gate in self.nn.text_cross_attn_gates.values()
                if gate.requires_grad
            ]
            if gate_hinge_list:
                losses["gate_penalty"] = torch.stack(gate_hinge_list).mean() * self._gate_penalty_weight

        self.log_losses(bs=bs, losses=losses, log_prefix=log_prefix, batch=batch)
        if self._diff_loss_weight != 1.0:
            train_loss = sum([
                torch.mean(losses[k]) * (
                    1.0 if ("alignment" in k or "com_separation" in k
                           or "distillation" in k or "adapter_ddp_anchor" in k
                           or "gate_penalty" in k or "delta_pres" in k)
                    else self._diff_loss_weight
                )
                for k in losses if "_justlog" not in k
            ])
        else:
            train_loss = sum([torch.mean(losses[k]) for k in losses if "_justlog" not in k])

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if not val_step:  # Don't log these for val step
            self.log_train_loss_n_prog_bar(bs, train_loss)
            self.update_n_log_flops(bs, n)
            self.update_n_log_nsamples_processed(bs)
            self.log_nparams()
            if self.global_step % self.trainer.log_every_n_steps == 0 and self.global_rank == 0:
                loss_parts = " | ".join(
                    f"{k}={torch.mean(losses[k]).detach().item():.4f}"
                    for k in sorted(losses) if "_justlog" not in k
                )
                logger.info(
                    f"[step {self.global_step}] loss={train_loss.detach().item():.4f} | {loss_parts}"
                )
                # Write to step log file (bypasses loguru/Lightning buffering)
                import time as _time
                _step_log = getattr(self, "_step_log_path", None)
                if _step_log is None:
                    import os, datetime as _dt
                    _out = os.environ.get("OUTPUT_ROOT", "/tmp")
                    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._step_log_path = os.path.join(_out, f"step_log_{_ts}.csv")
                    # Also create a symlink for monitoring scripts
                    _link = os.path.join(_out, "step_log.csv")
                    try:
                        os.remove(_link)
                    except FileNotFoundError:
                        pass
                    os.symlink(self._step_log_path, _link)
                    _header = "step,timestamp,loss,bb_ca,bb_ca_raw,local_latents,local_latents_raw\n"
                    with open(self._step_log_path, "w") as f:
                        f.write(_header)
                    _step_log = self._step_log_path
                _bb = torch.mean(losses["bb_ca"]).detach().item() if "bb_ca" in losses else 0.0
                _bb_raw = torch.mean(losses["bb_ca_mse_raw_justlog"]).detach().item() if "bb_ca_mse_raw_justlog" in losses else 0.0
                _ll = torch.mean(losses["local_latents"]).detach().item() if "local_latents" in losses else 0.0
                _ll_raw = torch.mean(losses["local_latents_mse_raw_justlog"]).detach().item() if "local_latents_mse_raw_justlog" in losses else 0.0
                with open(_step_log, "a") as f:
                    f.write(f"{self.global_step},{_time.time():.3f},{train_loss.detach().item():.4f},{_bb:.4f},{_bb_raw:.4f},{_ll:.4f},{_ll_raw:.4f}\n")
                    f.flush()
                _loss_val = train_loss.detach().item()
                if _loss_val != _loss_val:  # NaN check
                    import warnings
                    warnings.warn(f"[ALERT] NaN loss at step {self.global_step}!", stacklevel=2)
                elif _loss_val > 1000.0 and self.global_step > 100:
                    import warnings
                    warnings.warn(f"[ALERT] Loss explosion at step {self.global_step}: {_loss_val:.1f}", stacklevel=2)

            if hasattr(self.nn, "text_cross_attn_gates") and self.nn.text_cross_attn_gates is not None:
                for i, gate in self.nn.text_cross_attn_gates.items():
                    self.log(
                        f"{log_prefix}/gate_tanh/layer_{i}",
                        gate.tanh().item(),
                        on_step=True, on_epoch=False, prog_bar=False, logger=True,
                        sync_dist=True,
                    )
                    self.log(
                        f"{log_prefix}/gate_raw/layer_{i}",
                        gate.detach().item(),
                        on_step=True, on_epoch=False, prog_bar=False, logger=True,
                        sync_dist=True,
                    )

            if hasattr(self.nn, "text_cross_attn_adapter_gates") and self.nn.text_cross_attn_adapter_gates is not None:
                for i, gate in self.nn.text_cross_attn_adapter_gates.items():
                    self.log(
                        f"{log_prefix}/adapter_gate_tanh/layer_{i}",
                        gate.tanh().item(),
                        on_step=True, on_epoch=False, prog_bar=False, logger=True,
                        sync_dist=True,
                    )
                    self.log(
                        f"{log_prefix}/adapter_gate_raw/layer_{i}",
                        gate.detach().item(),
                        on_step=True, on_epoch=False, prog_bar=False, logger=True,
                        sync_dist=True,
                    )

            if (
                hasattr(self.nn, "text_cross_attn_output_scale_mode")
                and self.nn.text_cross_attn_output_scale_mode == "learnable"
                and self.nn.text_cross_attn_layers is not None
            ):
                for i, layer in self.nn.text_cross_attn_layers.items():
                    if not hasattr(layer, "output_gate"):
                        continue
                    g = layer.output_gate
                    self.log(
                        f"{log_prefix}/output_gate_mean/layer_{i}",
                        g.mean().item(),
                        on_step=True, on_epoch=False, prog_bar=False, logger=True,
                        sync_dist=True,
                    )
                    self.log(
                        f"{log_prefix}/output_gate_absmax/layer_{i}",
                        g.abs().max().item(),
                        on_step=True, on_epoch=False, prog_bar=False, logger=True,
                        sync_dist=True,
                    )

            # Alignment head diagnostics
            if self._align_enabled and self.alignment_head.layer_weights is not None:
                weights = self.alignment_head.layer_weights
                for li, w in enumerate(weights):
                    self.log(
                        f"{log_prefix}/align_layer_weight/{li}",
                        w.item(),
                        on_step=True, on_epoch=False, prog_bar=False, logger=True,
                        sync_dist=True,
                    )
                entropy = -(weights * weights.clamp(min=1e-8).log()).sum()
                self.log(
                    f"{log_prefix}/align_layer_weight_entropy",
                    entropy.item(),
                    on_step=True, on_epoch=False, prog_bar=False, logger=True,
                    sync_dist=True,
                )

        return train_loss

    def log_losses(
        self,
        bs: int,
        losses: dict[str, Float[torch.Tensor, "b"]],
        log_prefix: str,
        batch: dict,
    ):
        for k in losses:
            log_name = k[: -len("_justlog")] if k.endswith("_justlog") else k

            self.log(
                f"{log_prefix}/loss_{log_name}",
                torch.mean(losses[k]),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

        # Per-timestep-bin loss logging for FM modalities (10 bins of width 0.1).
        t_dict = batch.get("t", {})
        tbin_modes = [
            dm for dm in ("bb_ca", "local_latents")
            if dm in losses and dm in t_dict and log_prefix == "train"
        ]
        for dm in tbin_modes:
            t_dm = t_dict[dm].detach()
            loss_dm = losses[dm].detach()
            for bin_idx in range(10):
                lo = bin_idx * 0.1
                hi = lo + 0.1
                mask = (t_dm >= lo) & (t_dm < hi)
                if mask.any():
                    self.log(
                        f"{log_prefix}/loss_{dm}_tbin/{bin_idx}",
                        loss_dm[mask].mean(),
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        logger=True,
                        batch_size=int(mask.sum()),
                        sync_dist=True,
                        add_dataloader_idx=False,
                    )

        # Manual per-tbin accumulator (rank 0 only, bypasses Lightning).
        # Writes to tbin_log.csv at epoch boundaries. Immune to resume artifacts.
        if tbin_modes and self.global_rank == 0:
            if not hasattr(self, "_tbin_accum"):
                self._tbin_modes = [
                    mode for mode in tbin_modes
                    if (
                        t_dict[mode].dim() == 1
                        and losses[mode].dim() == 1
                        and t_dict[mode].shape == losses[mode].shape
                    )
                ]
                self._tbin_enabled = len(self._tbin_modes) > 0
                if self._tbin_enabled:
                    self._reset_tbin_accum()
                    self._tbin_epoch = self.current_epoch
                    import os, datetime as _dt
                    _out = os.environ.get("OUTPUT_ROOT", "/tmp")
                    _ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    self._tbin_log_path = os.path.join(_out, f"tbin_log_{_ts}.csv")
                    # Also create a symlink for monitoring scripts
                    _link = os.path.join(_out, "tbin_log.csv")
                    try:
                        os.remove(_link)
                    except FileNotFoundError:
                        pass
                    os.symlink(self._tbin_log_path, _link)
                    with open(self._tbin_log_path, "w") as f:
                        header = "epoch,step," + ",".join(
                            f"{mode}_tbin{i}_w,{mode}_tbin{i}_raw"
                            for mode in self._tbin_modes
                            for i in range(10)
                        ) + "\n"
                        f.write(header)
                    self._tbin_accum = True
                else:
                    self._tbin_accum = False

            if self._tbin_enabled:
                # Flush at epoch boundary
                if self.current_epoch != self._tbin_epoch:
                    self._flush_tbin_log(epoch=self._tbin_epoch)
                    self._reset_tbin_accum()
                    self._tbin_epoch = self.current_epoch

                # Accumulate (Python floats, no GPU memory held)
                for mode in self._tbin_modes:
                    t_vals = t_dict[mode].detach().cpu().tolist()
                    l_w_vals = losses[mode].detach().cpu().tolist()
                    raw_key = f"{mode}_mse_raw_justlog"
                    l_r_vals = (losses[raw_key].detach().cpu().tolist()
                                if raw_key in losses else [0.0] * len(t_vals))
                    for t_v, l_w, l_r in zip(t_vals, l_w_vals, l_r_vals):
                        idx = max(0, min(int(t_v * 10), 9))
                        self._tbin_sums_w[mode][idx] += l_w
                        self._tbin_sums_raw[mode][idx] += l_r
                        self._tbin_counts[mode][idx] += 1

        if self.cfg_exp.training.get("p_folding_n_inv_folding_iters", 0.0) > 0.0:
            # Log also for folding and inverse folding iters
            # divides by p_aux to account for the fact that for most steps loss will be just zero
            # (since need to sync across devices, etc, need to handle logging carefully)
            p_aux = self.cfg_exp.training["p_folding_n_inv_folding_iters"] / 2
            loss = torch.mean(losses[k])  # [b]

            f_inv_fold = batch["use_ca_coors_nm_feature"] * 1.0 / p_aux
            self.log(
                f"{log_prefix}_invfold_ca_iter/loss_{log_name}",
                loss * f_inv_fold,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

            f_fold = batch["use_residue_type_feature"] * 1.0 / p_aux
            self.log(
                f"{log_prefix}_fold_iter/loss_{log_name}",
                loss * f_fold,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

    def _reset_tbin_accum(self):
        self._tbin_sums_w = {mode: [0.0] * 10 for mode in self._tbin_modes}
        self._tbin_sums_raw = {mode: [0.0] * 10 for mode in self._tbin_modes}
        self._tbin_counts = {mode: [0] * 10 for mode in self._tbin_modes}

    def _flush_tbin_log(self, epoch: int):
        if not getattr(self, "_tbin_enabled", False):
            return
        avgs = []
        for mode in self._tbin_modes:
            for i in range(10):
                c = self._tbin_counts[mode][i]
                avgs.append(f"{self._tbin_sums_w[mode][i]/c:.6f}" if c > 0 else "")
                avgs.append(f"{self._tbin_sums_raw[mode][i]/c:.6f}" if c > 0 else "")
        with open(self._tbin_log_path, "a") as f:
            f.write(f"{epoch},{self.global_step}," + ",".join(avgs) + "\n")
            f.flush()

    def log_train_loss_n_prog_bar(self, b: int, train_loss: torch.Tensor):
        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=b,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def log_nparams(self):
        self.log(
            "scaling/nparams",
            self.nparams * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )  # constant line but ok, easy to compare # params

    def update_n_log_nsamples_processed(self, b: int):
        self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def update_n_log_flops(self, b: int, n: int):
        """
        Updates and logs flops, if available
        """
        try:
            nflops_step = self.nn.nflops_computer(b, n)  # nn should implement this function if we want to see nflops
        except Exception:
            nflops_step = None

        if nflops_step is not None:
            self.nflops = (
                self.nflops + nflops_step * self.trainer.world_size
            )  # Times number of processes so it logs sum across devices
            self.log(
                "scaling/nflops",
                self.nflops * 1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int):
        """Validation step dispatching to length-based generation or data-based loss.

        Args:
            batch: batch from dataset
            batch_idx: batch index (unused)
            dataloader_idx: 0 for length dataloader (generation), 1 for data dataloader (loss)
        """
        if dataloader_idx == 0:
            self.validation_step_lens(batch, batch_idx)
        elif dataloader_idx == 1:
            self.validation_step_data(batch, batch_idx)
        else:
            raise OSError(f"Validation dataloader with index {dataloader_idx} not recognized")

    def validation_step_data(self, batch: dict, batch_idx: int):
        """Evaluates the training loss on validation data."""
        with torch.no_grad():
            loss = self.training_step(batch, batch_idx=-1)
            self.validation_output_data.append(loss.item())

    def validation_step_lens(self, batch: dict, batch_idx: int):
        """
        Generates samples and saves samples in the self.validatoin_output list. Each sample is stored as the atom37
        coordinates, of shape [n, 37, 3]. The variable self.validation_output is thus a list of tensors of shape
        [n, 37, 3].

        Args:
            batch: data batch, contains no data, but the info of the samples to generate (nsamples, nres, dt).

        Returns:
            Nothing, just stores the samples in the list.
        """
        sampling_args = copy.deepcopy(self.cfg_exp.generation.args)
        cath_code = (
            extract_cath_code_from_batch(batch) if sampling_args.fold_cond else None
        )  # When using unconditional model, don't use cath_code
        del sampling_args.fold_cond
        del sampling_args.ag_ckpt_path
        assert sampling_args.ag_ratio == 0.0, "Should turn off autoguidance for validation"

        with torch.no_grad():
            for val_mode in self.cfg_exp.generation.model:
                # fn_predict_for_sampling = partial(
                #     self.predict_for_sampling, n_recycle=0
                # )
                fn_predict_for_sampling = self.predict_for_sampling
                gen_samples = self.fm.full_simulation(
                    predict_for_sampling=fn_predict_for_sampling,
                    batch=batch,
                    nsteps=400,
                    nsamples=batch["nsamples"],
                    n=batch["nres"],
                    self_cond=False,
                    sampling_model_args=self.cfg_exp.generation.model[val_mode],
                    device=self.device,
                )
                # Dict with the data_modes as keys, and values with batch shape b

                # Format the generated samples back to proteins
                sample_prots = sample_formatting(
                    x=gen_samples,
                    # `text_cond` rides along so the AE DECODER can be text
                    # conditioned. It is a no-op unless the decoder config lists
                    # "text_cond" in feats_cond_seq; a stock AE ignores the key.
                    extra_info={
                        "mask": batch["mask"],
                        **{k: batch[k] for k in
                           ("text_cond", "text_tokens", "text_tokens_mask")
                           if k in batch},
                    },
                    ret_mode="coors37_n_aatype",
                    data_modes=list(self.cfg_exp.product_flowmatcher),
                    autoencoder=getattr(self, "autoencoder", None),
                )
                # Dict with keys `coors` (a37), `residue_type`, and `mask`,
                # shapes [b, n, 37, 3], [b, n], [b, n]

                generation_list = []
                for i in range(sample_prots["coors"].shape[0]):
                    generation_list.append(
                        (sample_prots["coors"][i], sample_prots["residue_type"][i])
                    )  # Tuple (coors [n, 37, 3], aatype [n])

                if val_mode not in self.validation_output_lens:
                    self.validation_output_lens[val_mode] = []
                self.validation_output_lens[val_mode] += generation_list

    def on_train_epoch_end(self):
        """Flush per-tbin accumulator at epoch end (rank 0 only)."""
        if getattr(self, "_tbin_enabled", False) and self.global_rank == 0:
            has_counts = any(
                c > 0
                for mode_counts in self._tbin_counts.values()
                for c in mode_counts
            )
            if has_counts:
                self._flush_tbin_log(epoch=self.current_epoch)
            self._reset_tbin_accum()
            self._tbin_epoch = self.current_epoch + 1

    def on_validation_epoch_end(self):
        """Process validation results at epoch end."""
        self.on_validation_epoch_end_data()
        # TODO: Re-enable length-based generation validation once refactored.
        # Disabled because it is expensive (generates full PDB samples + runs
        # structural metrics every val epoch) and currently under rework for
        # the new group-based collation.  Training loss validation still runs.
        # self.on_validation_epoch_end_lens()

    def on_validation_epoch_end_data(self):
        self.validation_output_data = []

    def on_validation_epoch_end_lens(self):
        """
        Generates PDB files from produced samples and computes metrics.
        It does this for all sampling modes considered.
        """
        # Save structures to pdb files
        for val_mode in self.validation_output_lens:
            paths = []
            len_tracker = {}
            for i, (coors_atom37, residue_type) in enumerate(self.validation_output_lens[val_mode]):
                n = coors_atom37.shape[-3]
                len_tracker[n] = len_tracker.get(n, 0)
                len_tracker[n] += 1
                name = (
                    f"epoch_{self.current_epoch}_bid_n_{n}_num_{len_tracker[n]}_rank_{self.global_rank}_{val_mode}.pdb"
                )
                full_path = os.path.join(self.val_path_tmp, name)
                if not os.path.exists(self.val_path_tmp):
                    create_dir(self.val_path_tmp)
                try:
                    write_prot_to_pdb(
                        prot_pos=coors_atom37.float().detach().cpu().numpy(),
                        aatype=residue_type.detach().cpu().numpy(),
                        file_path=full_path,
                        overwrite=True,
                        no_indexing=True,
                    )
                    paths.append(full_path)
                except Exception as e:
                    logger.error(
                        f"[Global rank: {self.global_rank}]: Failed to write protein to PDB on validation: {e}"
                    )

            # Subset of paths for non new metrics
            paths_subset = paths.copy()
            random.shuffle(paths_subset)
            paths_subset = paths_subset[:40]

            # Clean up
            clean_validation_files(paths)
        self.validation_output_lens = {}

    def configure_inference(self, inf_cfg, nn_ag):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag

    def generate(self, batch: dict) -> dict:
        """
        Runs a single generation pass with the current configuration.

        Args:
            batch: Data batch containing generation parameters. Must contain 'mask' tensor.

        Returns:
            gen_samples: Dictionary with generated samples for each data mode
        """
        self_cond = self.inf_cfg.args.self_cond
        nsteps = self.inf_cfg.args.nsteps
        guidance_w = self.inf_cfg.args.get("guidance_w", 1.0)
        ag_ratio = self.inf_cfg.args.get("ag_ratio", 0.0)
        guidance_alpha = self.inf_cfg.args.get("guidance_alpha", 0.0)
        contrastive_w = self.inf_cfg.args.get("contrastive_w", 0.0)

        fn_predict_for_sampling = partial(self.predict_for_sampling, n_recycle=self.inf_cfg.get("n_recycle", 0))

        # Derive nsamples and n from mask shape
        mask = batch["mask"]
        nsamples, n = mask.shape

        gen_samples = self.fm.full_simulation(
            batch=batch,
            predict_for_sampling=fn_predict_for_sampling,
            nsteps=nsteps,
            nsamples=nsamples,
            n=n,
            self_cond=self_cond,
            sampling_model_args=self.inf_cfg.model,
            device=self.device,
            guidance_w=guidance_w,
            ag_ratio=ag_ratio,
            guidance_alpha=guidance_alpha,
            contrastive_w=contrastive_w,
        )

        return gen_samples

    # ------------------------------------------------------------------
    # predict_step helpers
    # ------------------------------------------------------------------

    def _get_search_instance(self):
        """Return cached search instance, re-creating only if algorithm changed."""
        search_algorithm = getattr(self.inf_cfg, "search", {}).get("algorithm", "single-pass")
        if (
            not hasattr(self, "_search_instance")
            or self._search_instance is None
            or getattr(self, "_search_algorithm", None) != search_algorithm
        ):
            self._search_instance = instantiate_search(self, self.inf_cfg, search_algorithm)
            self._search_algorithm = search_algorithm
        return self._search_instance

    def _refinement_enabled(self) -> bool:
        ref_cfg = getattr(self.inf_cfg, "refinement", {})
        return bool(ref_cfg.get("algorithm", None)) if ref_cfg else False

    def _apply_refinement(self, final_prots, lookahead_prots):
        """Refine samples and optionally save pre-refinement copies.

        Only call when ``_refinement_enabled()`` is True.

        Returns (final_prots, lookahead_prots, unrefined_final, unrefined_lookahead).

        NOTE: SequenceHallucination.refine() accepts target_hotspot_mask but
        the original hardcoded hotspot=None.  When hotspot-guided refinement
        is needed, pass expand_hotspot_mask(...) here.
        """
        ref_cfg = self.inf_cfg.refinement
        algorithm = ref_cfg.get("algorithm")

        if not hasattr(self, "_refinement_instance") or self._refinement_instance is None:
            self._refinement_instance = instantiate_refinement(self, self.inf_cfg, algorithm)

        refine_targets = ref_cfg.get("refine_targets", "final")
        save_pre = ref_cfg.get("save_pre_refinement", "none")

        unrefined_final = clone_sample_dict(final_prots) if save_pre in ("final", "all") else None
        unrefined_lookahead = None
        if save_pre == "all" and refine_targets == "all" and lookahead_prots is not None:
            unrefined_lookahead = clone_sample_dict(lookahead_prots)

        final_prots = self._refinement_instance.refine(final_prots)
        if refine_targets == "all" and lookahead_prots is not None:
            lookahead_prots = self._refinement_instance.refine(lookahead_prots)

        return final_prots, lookahead_prots, unrefined_final, unrefined_lookahead

    def _append_unrefined_samples(
        self,
        sample_prots,
        unrefined_final,
        unrefined_lookahead,
        expanded_hotspot_mask,
        ligand,
    ):
        """Score and append pre-refinement copies to the output dict."""
        if unrefined_final is not None:
            unrefined_final_rewards = None
            if self.reward_model is not None:
                unrefined_final_rewards = compute_reward_from_samples(
                    self.reward_model,
                    unrefined_final,
                    expanded_hotspot_mask,
                    ligand,
                )
            append_samples(
                sample_prots,
                unrefined_final,
                unrefined_final_rewards,
                "final_unrefined",
            )
        if unrefined_lookahead is not None:
            append_samples(
                sample_prots,
                unrefined_lookahead,
                unrefined_lookahead.get("rewards"),
                "lookahead_unrefined",
            )

    # ------------------------------------------------------------------
    # predict_step
    # ------------------------------------------------------------------

    @torch.inference_mode(mode=True)
    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        """Run search → (optional) refinement → reward scoring → combine.

        Args:
            batch: Must contain ``mask`` tensor [batch_size, n_residues].

        Returns:
            Dict with ``coors``, ``residue_type``, ``mask``, ``rewards``,
            ``sample_type``, and optionally ``chain_index`` / ``metadata_tag``.
        """
        if "mask" not in batch:
            raise ValueError("Batch must contain 'mask' tensor")

        # Text conditioning for inference — encode once, reused across
        # all sampling steps via the batch dict.
        if self.text_encoder is not None and "text" in batch:
            pooled_cond, token_cond, token_mask, _ = self.text_encoder(
                batch["text"], device=batch["mask"].device, training=False
            )
            if pooled_cond is not None:
                batch["text_cond"] = pooled_cond
            if token_cond is not None:
                batch["text_tokens"] = token_cond
                batch["text_tokens_mask"] = token_mask

            # Contrastive guidance: encode baseline prompts (y⁻) if present
            contrastive_w = (
                self.inf_cfg.args.get("contrastive_w", 0.0) if self.inf_cfg else 0.0
            )
            if "baseline_text" in batch and any(t.strip() for t in batch["baseline_text"]):
                pooled_bl, tokens_bl, tokens_mask_bl, _ = self.text_encoder(
                    batch["baseline_text"], device=batch["mask"].device, training=False
                )
                if pooled_bl is not None:
                    batch["text_cond_baseline"] = pooled_bl
                if tokens_bl is not None:
                    batch["text_tokens_baseline"] = tokens_bl
                    batch["text_tokens_mask_baseline"] = tokens_mask_bl
            elif contrastive_w != 0.0:
                logger.warning(
                    "contrastive_w={} requested but no baseline_description in CSV; "
                    "contrastive guidance disabled for this batch.",
                    contrastive_w,
                )

        if not hasattr(self, "reward_model") or self.reward_model is None:
            self.reward_model = initialize_reward_model(self.inf_cfg)

        # ---- Search ----
        search_result = self._get_search_instance().search(batch)
        final_prots = search_result["final"]
        lookahead_prots = search_result.get("lookahead")

        # ---- Refinement (only when configured) ----
        unrefined_final = None
        unrefined_lookahead = None
        if self._refinement_enabled():
            final_prots, lookahead_prots, unrefined_final, unrefined_lookahead = self._apply_refinement(
                final_prots, lookahead_prots
            )

        # ---- Score finals ----
        # Lookaheads are scored during search (tile layout, modulo correct).
        # Finals are scored here.  All search algorithms output finals in
        # grouped layout (all beams/replicas for sample 0 first, then 1, …).
        #
        # BUG FIX: the original passed raw [nsamples] hotspot_mask and used
        # i % nsamples which is wrong for grouped layout.
        # expand_hotspot_mask uses repeat_interleave to match.
        nsamples = batch["mask"].shape[0]
        hotspot_mask = batch.get("target_hotspot_mask")
        ligand = getattr(self, "ligand", None)
        expanded_hotspot_mask = expand_hotspot_mask(
            hotspot_mask,
            final_prots["coors"].shape[0],
            nsamples,
        )

        final_rewards = None
        if self.reward_model is not None:
            final_rewards = compute_reward_from_samples(
                self.reward_model,
                final_prots,
                expanded_hotspot_mask,
                ligand,
            )

        # ---- Combine output ----
        sample_prots = combine_lookahead_and_final(
            lookahead=lookahead_prots,
            final=final_prots,
            final_rewards=final_rewards,
        )

        if unrefined_final is not None or unrefined_lookahead is not None:
            self._append_unrefined_samples(
                sample_prots,
                unrefined_final,
                unrefined_lookahead,
                expanded_hotspot_mask,
                ligand,
            )

        if "metadata_tag" in batch:
            sample_prots["metadata_tag"] = batch["metadata_tag"]

        return sample_prots
