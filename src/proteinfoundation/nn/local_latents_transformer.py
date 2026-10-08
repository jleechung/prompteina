# Modified by the Prompteina authors in 2026.
# Changes are licensed under Apache 2.0; see LICENSE and NOTICE.
import torch
from openfold.np.residue_constants import RESTYPE_ATOM37_MASK

from proteinfoundation.nn.feature_factory.concat_feature_factory import ConcatFeaturesFactory
from proteinfoundation.nn.feature_factory.concat_pair_feature_factory import ConcatPairFeaturesFactory
from proteinfoundation.nn.feature_factory.feature_factory import FeatureFactory
from proteinfoundation.nn.modules.attn_n_transition import MultiheadAttnAndTransition, MultiheadCrossAttnAndTransition
from proteinfoundation.nn.modules.pair_update import PairReprUpdate
from proteinfoundation.nn.modules.seq_transition_af3 import Transition
from proteinfoundation.nn.modules.text_cross_attn import TextCrossAttentionADALN
from proteinfoundation.nn.protein_transformer import PairReprBuilder
from proteinfoundation.utils.tensor_utils import compact_concat_indices


def get_atom_mask(device: torch.device = None):
    return torch.from_numpy(RESTYPE_ATOM37_MASK).to(dtype=torch.bool, device=device)  # [21, 37]


class LocalLatentsTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.

    Now supports concat features for motif and/or target conditioning.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super().__init__()
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_pair_repr = kwargs["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["use_tri_mult"]
        self.use_tri_attn = kwargs.get("use_tri_attn", False)
        self.use_qkln = kwargs["use_qkln"]
        self.output_param = kwargs["output_parameterization"]

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # Concat features for motif and/or target
        concat_config = kwargs.get("concat_features", {})
        self.use_concat = (
            concat_config.get("enable_motif", False)
            or concat_config.get("enable_target", False)
            or concat_config.get("enable_ligand", False)
        )

        if self.use_concat:
            self.concat_factory = ConcatFeaturesFactory(
                enable_motif=concat_config.get("enable_motif", False),
                enable_target=concat_config.get("enable_target", False),
                enable_ligand=concat_config.get("enable_ligand", False),
                dim_feats_out=kwargs["token_dim"],
                use_ln_out=False,
                **kwargs,
            )

            # Check for advanced pair representation mode
            self.use_advanced_pair = (
                (concat_config.get("enable_motif", False) and concat_config.get("motif_pair_features", False))
                or (concat_config.get("enable_target", False) and concat_config.get("target_pair_features", False))
                or (concat_config.get("enable_ligand", False) and concat_config.get("ligand_pair_features", False))
            )

            if self.use_advanced_pair:
                # Initialize pair features factory for extended pair representations
                self.concat_pair_factory = ConcatPairFeaturesFactory(
                    enable_motif=concat_config.get("enable_motif", False),
                    enable_target=concat_config.get("enable_target", False),
                    enable_ligand=concat_config.get("enable_ligand", False),
                    **kwargs,
                )
                from loguru import logger

                logger.info("Enabled advanced pair representation with cross-sequence features")
        else:
            self.concat_factory = None
            self.use_advanced_pair = False

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        self.use_target_cross_attn = kwargs.get("use_target_cross_attn", False)
        if self.use_target_cross_attn:
            assert not self.use_concat, "use_target_cross_attn requires use_concat to be False"
            self.target_factory = FeatureFactory(
                feats=kwargs["feats_target"],
                dim_feats_out=kwargs["token_dim"],
                use_ln_out=False,
                mode="target",
                **kwargs,
            )
            self.target2binder_cross_attention_layer = torch.nn.ModuleList(
                [
                    MultiheadCrossAttnAndTransition(
                        dim_token_a=kwargs["token_dim"],
                        dim_token_b=kwargs["token_dim"],
                        nheads=kwargs["nheads"],
                        dim_cond=kwargs["dim_cond"],
                        residual_mha=True,
                        residual_transition=True,
                        use_qkln=self.use_qkln,
                    )
                    for _ in range(self.nlayers)
                ]
            )

        # Text cross-attention layers (token path)
        self.use_text_cross_attn = kwargs.get("use_text_cross_attn", False)
        self.text_cross_attn_layer_interval = kwargs.get("text_cross_attn_layer_interval", 1)
        self.force_text_attn_maps = kwargs.get("force_text_attn_maps", False)
        self.side_compute_attn_weights = kwargs.get("side_compute_attn_weights", False)

        self.use_text_cross_attn_gate = kwargs.get("use_text_cross_attn_gate", False)
        self.mask_target_cross_attn_delta = kwargs.get("mask_target_cross_attn_delta", False)
        self.detach_cross_attn_delta = kwargs.get("detach_cross_attn_delta", False)
        self.detach_target_delta_for_alignment = kwargs.get("detach_target_delta_for_alignment", False)
        self.text_cross_attn_output_scale_mode = kwargs.get(
            "text_cross_attn_output_scale_mode", "adaln_zero",
        )
        self.attention_type = kwargs.get("attention_type", "naive")

        if self.use_text_cross_attn:
            cross_attn_layer_indices = [
                i for i in range(self.nlayers)
                if i % self.text_cross_attn_layer_interval == 0
            ]
            self.text_cross_attn_layers = torch.nn.ModuleDict({
                str(i): TextCrossAttentionADALN(
                    d_model=self.token_dim,
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    text_cross_attn_output_scale_mode=self.text_cross_attn_output_scale_mode,
                )
                for i in cross_attn_layer_indices
            })
            if self.side_compute_attn_weights:
                for layer in self.text_cross_attn_layers.values():
                    layer.side_compute_attn_weights = True

            # Optional tanh scalar gate on cross-attention residual (SteerViT-style)
            if self.use_text_cross_attn_gate:
                self.text_cross_attn_gates = torch.nn.ParameterDict({
                    str(i): torch.nn.Parameter(torch.tensor(0.0))
                    for i in cross_attn_layer_indices
                })
            else:
                self.text_cross_attn_gates = None
        else:
            self.text_cross_attn_layers = None
            self.text_cross_attn_gates = None

        # Parallel adapter cross-attention (zero-init gates, for alignment training
        # on parents with large gates where modifying existing cross-attn disrupts
        # the denoiser). Off by default — existing behavior unchanged.
        self.use_text_cross_attn_adapter = kwargs.get("use_text_cross_attn_adapter", False)
        if self.use_text_cross_attn and self.use_text_cross_attn_adapter:
            self.text_cross_attn_adapter_layers = torch.nn.ModuleDict({
                str(i): TextCrossAttentionADALN(
                    d_model=self.token_dim,
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    text_cross_attn_output_scale_mode=self.text_cross_attn_output_scale_mode,
                )
                for i in cross_attn_layer_indices
            })
            self.text_cross_attn_adapter_gates = torch.nn.ParameterDict({
                str(i): torch.nn.Parameter(torch.tensor(0.0))
                for i in cross_attn_layer_indices
            })
        else:
            self.text_cross_attn_adapter_layers = None
            self.text_cross_attn_adapter_gates = None

        # To get pair representation
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr"],
            feats_cond=kwargs["feats_pair_cond"],
            dim_feats_out=kwargs["pair_repr_dim"],
            dim_cond_pair=kwargs["dim_cond"],
            **kwargs,
        )

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=self.use_qkln,
                    attention_type=self.attention_type,
                )
                for _ in range(self.nlayers)
            ]
        )

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                            use_tri_attn=self.use_tri_attn,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        self.local_latents_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
        )
        self.ca_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 3, bias=False),
        )

        # self.linear_out = torch.nn.Sequential(
        #     torch.nn.LayerNorm(self.token_dim),
        #     torch.nn.Linear(self.token_dim, kwargs["latent_dim"] + 3, bias=False),
        # )

    # @torch.compile
    def forward(self, input: dict) -> dict[str, dict[str, torch.Tensor]]:
        """
        Runs the network.

        Args:
            input: {
                # Sampling and training
                "x_t": Dict[str, torch.Tensor[b, n, dim]]
                "t": Dict[str, torch.Tensor[b]]
                "mask": boolean torch.Tensor[b, n]

                # Only training (other batch elements)
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "residue_mask": boolean torch.Tensor(b, n)
                ...
            }

        Returns:
            Dictionary:
            {
                "coors_nm": all atom coordinates, shape [b, n, 37, 3]
                "seq_logits": logits for the residue types, shape [b, n, 20]
                "residue_mask": boolean [b, n]
                "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
                "atom_mask": boolean [b, n, 37], atom37 mask corresponding to aatype_max
            }
        """
        mask = input["mask"]  # [b, n] boolean
        orig_mask = mask.clone()  # [b, n] boolean

        # Conditioning variables
        c = self.cond_factory(input)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Initial sequence representation from features
        seq_f_repr = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seq_f_repr * mask[..., None]  # [b, n, token_dim]
        # Store original dimensions
        b, n_orig, _ = seqs.shape
        # Extend sequence representation with concat features if available
        concat_layout = None
        if self.use_concat:
            seqs, mask = self.concat_factory(
                input, seqs, mask
            )  # [b, n_extended, token_dim], [b, n_extended], [b, n_extended]
            n_extended = seqs.shape[1]
            n_concat = n_extended - n_orig

            if "seq_target_mask" in input:
                concat_mask = input["seq_target_mask"].bool()
                concat_layout = compact_concat_indices(orig_mask, concat_mask, pad_len=n_extended)
                input["binder_idx_ext"] = concat_layout["binder_idx_ext"]
                input["target_idx_ext"] = concat_layout["target_idx_ext"]
                input["binder_mask_ext"] = concat_layout["binder_mask_ext"]
                input["target_mask_ext"] = concat_layout["target_mask_ext"]
                input["pair_binder_target_mask"] = concat_layout["pair_binder_target_mask"]
                input["pair_target_binder_mask"] = concat_layout["pair_target_binder_mask"]

            # Extend conditioning with zeros for concat features
            if n_concat > 0:
                zero_cond = torch.zeros(b, n_concat, c.shape[-1], device=seqs.device)
                c = torch.cat([c, zero_cond], dim=1)  # [b, n_extended, dim_cond]
        else:
            n_extended = n_orig
            n_concat = 0

        # Compute pair representation
        if self.use_concat and self.use_advanced_pair:  # and n_concat > 0:
            # Advanced mode - compute pair features for extended sequence
            pair_rep = self.pair_repr_builder(input)  # [b, n_orig, n_orig, pair_dim]
            pair_rep = self.concat_pair_factory(input, pair_rep, orig_mask)  # [b, n_extended, n_extended, pair_dim]
        else:
            # Simple mode - compute pair representation for original sequence only
            pair_rep = self.pair_repr_builder(input)  # [b, n_orig, n_orig, pair_dim]

            # Extend pair representation with zeros if we have concat features
            if n_concat > 0:
                dim_pair = pair_rep.shape[-1]

                # Pad first dimension: [b, n_orig, n_orig, d] -> [b, n_extended, n_orig, d]
                zero_pad_1 = torch.zeros(b, n_concat, n_orig, dim_pair, device=seqs.device)
                pair_rep = torch.cat([pair_rep, zero_pad_1], dim=1)

                # Pad second dimension: [b, n_extended, n_orig, d] -> [b, n_extended, n_extended, d]
                zero_pad_2 = torch.zeros(b, n_extended, n_concat, dim_pair, device=seqs.device)
                pair_rep = torch.cat([pair_rep, zero_pad_2], dim=2)

        if self.use_target_cross_attn:
            target_rep = self.target_factory(input)  # [b, n_target, token_dim]
            target_mask = input["seq_target_mask"]
            target_rep = target_rep * target_mask[..., None]  # [b, n_target, token_dim]

        # Text tokens for cross-attention (if present in batch)
        text_tokens = input.get("text_tokens", None)         # [b, L, token_dim] or None
        text_tokens_mask = input.get("text_tokens_mask", None)  # [b, L] bool or None

        # Collect attention maps for grounding losses or inference visualization.
        # Note: "grounding_mask" may be present for the alignment head without
        # needing attention maps. Use explicit flag to avoid disabling flash attn.
        return_text_attn_maps = (
            input.get("collect_text_attn_maps", False)
            or self.force_text_attn_maps
        )
        text_attn_maps = [] if return_text_attn_maps else None

        # Opt-in intermediate collection for alignment head
        collect_block_hidden = input.get("collect_block_hidden", False)
        collect_block_hidden_last = input.get("collect_block_hidden_last", False)
        collect_cross_attn_deltas = input.get("collect_cross_attn_deltas", False)
        collect_cross_attn_deltas_extended = input.get("collect_cross_attn_deltas_extended", False)
        collect_pair_rep = input.get("collect_pair_rep", False)
        block_hidden_list = [] if collect_block_hidden else None
        cross_attn_delta_list = [] if collect_cross_attn_deltas else None
        cross_attn_delta_extended_list = [] if collect_cross_attn_deltas_extended else None

        # Run trunk
        for i in range(self.nlayers):
            # Text cross-attention (before main MHSA, matching la-proteina ordering)
            if (
                self.use_text_cross_attn
                and self.text_cross_attn_layers is not None
                and str(i) in self.text_cross_attn_layers
                and text_tokens is not None
                and text_tokens_mask is not None
            ):
                result = self.text_cross_attn_layers[str(i)](
                    seqs, text_tokens, text_tokens_mask, c, mask,
                    return_attn_weights=return_text_attn_maps,
                )
                if return_text_attn_maps:
                    delta, attn_w = result
                    text_attn_maps.append(attn_w[:, :n_orig, :])
                else:
                    delta = result
                if self.text_cross_attn_gates is not None:
                    delta = self.text_cross_attn_gates[str(i)].tanh() * delta
                # Parallel adapter cross-attention (additive, zero-init gates).
                if (
                    self.text_cross_attn_adapter_layers is not None
                    and str(i) in self.text_cross_attn_adapter_layers
                    and not input.get("disable_adapter", False)
                ):
                    adapter_delta = self.text_cross_attn_adapter_layers[str(i)](
                        seqs, text_tokens, text_tokens_mask, c, mask,
                    )
                    adapter_delta = self.text_cross_attn_adapter_gates[str(i)].tanh() * adapter_delta
                    delta = delta + adapter_delta
                # Collect deltas for alignment head BEFORE masking.
                # When detach_target_delta_for_alignment=true, the alignment
                # head sees real target-side deltas (for bilateral interface
                # grounding) but only binder-side deltas produce gradients
                # through Q/K/V.
                if cross_attn_delta_list is not None:
                    cross_attn_delta_list.append(delta[:, :n_orig, :])
                if cross_attn_delta_extended_list is not None:
                    if self.detach_target_delta_for_alignment and n_concat > 0:
                        if concat_layout is not None:
                            target_mask = concat_layout["target_mask_ext"]
                            delta_for_align = torch.where(
                                target_mask[..., None],
                                delta.detach(),
                                delta,
                            )
                        else:
                            delta_for_align = torch.cat(
                                [delta[:, :n_orig], delta[:, n_orig:].detach()],
                                dim=1,
                            )
                        cross_attn_delta_extended_list.append(delta_for_align)
                    else:
                        cross_attn_delta_extended_list.append(delta)
                # Mask target delta from denoiser sequence.
                if self.mask_target_cross_attn_delta and n_concat > 0:
                    if concat_layout is not None:
                        delta = delta.masked_fill(concat_layout["target_mask_ext"][..., None], 0.0)
                    else:
                        delta = delta.clone()
                        delta[:, n_orig:, :] = 0.0
                if self.detach_cross_attn_delta:
                    seqs = seqs + delta.detach()
                else:
                    seqs = seqs + delta

            if self.use_target_cross_attn:
                seqs = self.target2binder_cross_attention_layer[i](
                    seqs, target_rep, c, mask, target_mask
                )  # [b, n_extended, token_dim], [b, n_target, token_dim]

            seqs = self.transformer_layers[i](
                seqs, pair_rep, c, mask
            )  # [b, n_extended, token_dim]

            if block_hidden_list is not None:
                block_hidden_list.append(seqs[:, :n_orig, :])

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](
                            seqs, pair_rep, mask
                        )  # [b, n_extended, n_extended, pair_dim]

        # Get outputs
        local_latents_out = self.local_latents_linear(seqs) * mask[..., None]  # [b, n_extended, latent_dim]
        ca_nm_out = self.ca_linear(seqs) * mask[..., None]  # [b, n_extended, 3]

        # Trim back to original sequence length (remove concat features) if we extended
        if n_concat > 0:
            local_latents_out = local_latents_out[:, :n_orig, :] * orig_mask[:, :, None]

            ca_nm_out = ca_nm_out[:, :n_orig, :] * orig_mask[:, :, None]

        nn_out = {}
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: ca_nm_out}
        nn_out["local_latents"] = {self.output_param["local_latents"]: local_latents_out}
        if text_attn_maps:
            nn_out["text_cross_attn_maps"] = text_attn_maps

        # Alignment head intermediates (opt-in, only when alignment head is active)
        if block_hidden_list:
            nn_out["block_hidden"] = torch.stack(block_hidden_list, dim=1)  # [b, nlayers, n_orig, d]
        if collect_block_hidden_last and not collect_block_hidden:
            # Cheap path: just the final layer, no per-layer stacking
            nn_out["block_hidden_last"] = seqs[:, :n_orig, :]  # [b, n_orig, d]
        if cross_attn_delta_list:
            nn_out["cross_attn_deltas"] = torch.stack(cross_attn_delta_list, dim=1)  # [b, n_ca, n_orig, d]
        if cross_attn_delta_extended_list:
            nn_out["cross_attn_deltas_extended"] = torch.stack(cross_attn_delta_extended_list, dim=1)
        if collect_pair_rep:
            nn_out["pair_rep"] = pair_rep

        return nn_out
