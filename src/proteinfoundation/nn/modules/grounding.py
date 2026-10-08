"""Grounding utilities for text-to-residue alignment.

``build_grounding_masks`` is the shared entry point — used by both the
alignment head (active) and the cross-attention grounding loss (deprecated).

Cross-attention grounding losses (compute_token_loss, compute_pixel_loss,
compute_grounding_loss) are DEPRECATED.  They require need_weights=True on
cross-attention layers which disables flash attention and is too slow for
production training.  Use ``AlignmentHead`` (alignment_head.py) instead,
which learns a dot-product projection head and does not need attention maps.

References:
  - TokenCompose (Wang et al., 2023): token-level + pixel-level attention loss
  - Dataset: text-struct-data alignment annotations (text_chars → residues)
"""

from __future__ import annotations

import logging
import warnings

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

INTERFACE_RESIDUE_FACTS = {
    "interface_residues",
    "interface_hbonds",
    "interface_salt_bridges",
    "interface_hydrophobic_contacts",
}

INTERFACE_PAIR_FACTS = {
    "interface_hbonds",
    "interface_salt_bridges",
    "interface_hydrophobic_contacts",
}


class GroundingLayerWeights(nn.Module):
    """Learnable per-layer weights for combining cross-attention grounding losses.

    Produces softmax-normalized weights so each layer's contribution is learned
    rather than uniform.  Initialized to equal weights (all logits = 0).

    Args:
        n_layers: Number of cross-attention layers to weight.
    """

    def __init__(self, n_layers: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_layers))

    def forward(self) -> Tensor:
        """Returns softmax-normalized layer weights [n_layers]."""
        return torch.softmax(self.logits, dim=0)


def build_grounding_masks(
    texts: list[str],
    alignments_list: list[list[dict]],
    modelled_positions_list: list[list[int]],
    tokenizer,
    n_residues: int,
    device: torch.device,
    use_chat_template: bool = False,
    res_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Build binary grounding masks from alignment annotations.

    Args:
        texts: List of description strings, length B.
        alignments_list: Per-sample list of alignment dicts, each with
            ``text_chars`` (char span) and ``residues`` ([[chain, pos], ...]).
        modelled_positions_list: Per-sample list of modeled label_seq_id
            values.  Index i in this list corresponds to tensor index i.
        tokenizer: HuggingFace tokenizer (must support return_offsets_mapping).
        n_residues: Padded residue dimension (from batch mask).
        device: Target device.
        use_chat_template: If True, apply chat template before tokenizing
            (must match the text encoder's tokenization).
        res_mask: Optional [B, N_res] bool mask.  When provided, each sample's
            ``modelled_positions`` length is validated against the actual
            residue count.  Samples with mismatches are skipped (mask stays
            zero) to avoid silent index misalignment between the annotation
            pipeline (gemmi) and the data loader (atomworks/biotite).

    Returns:
        grounding_mask: [B, L_txt, N_res] float tensor (1.0 where grounded).
        token_has_mask: [B, L_txt] bool tensor (True for tokens with any annotation).
        stats: dict with ``n_samples``, ``n_skipped_mismatch``, ``n_skipped_empty``.
    """
    batch_size = len(texts)
    n_skipped_mismatch = 0
    n_skipped_empty = 0

    # Apply chat template if the encoder does (must match _encode_tokens)
    tokenize_texts = texts
    # Per-sample char offset from chat template prefix
    char_offsets = [0] * batch_size
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        tokenize_texts = []
        for b_idx, t in enumerate(texts):
            wrapped = tokenizer.apply_chat_template(
                [{"role": "user", "content": t}],
                tokenize=False,
                add_generation_prompt=False,
            )
            # Find where the raw text starts in the wrapped string
            idx = wrapped.find(t)
            if idx < 0:
                logger.warning(
                    "Chat template modified text content; "
                    "grounding char offsets may be incorrect"
                )
                idx = 0
            char_offsets[b_idx] = idx
            tokenize_texts.append(wrapped)

    # Tokenize all texts to get offset mappings
    enc = tokenizer(
        tokenize_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offset_mapping = enc["offset_mapping"]  # [B, L_txt, 2]
    n_txt = offset_mapping.shape[1]

    grounding_mask = torch.zeros(batch_size, n_txt, n_residues, device=device)
    token_has_mask = torch.zeros(batch_size, n_txt, dtype=torch.bool, device=device)

    for b in range(batch_size):
        alignments = alignments_list[b]
        mp = modelled_positions_list[b]
        if not alignments or not mp:
            n_skipped_empty += 1
            continue

        # Validate that modelled_positions length matches actual residue count.
        # Mismatches arise when gemmi and atomworks/biotite disagree on which
        # residues to include (modified residues, unresolved atoms, etc.).
        # In that case indices after the divergence point are wrong — skip.
        if res_mask is not None:
            n_actual = int(res_mask[b].sum().item())
            if len(mp) != n_actual:
                logger.warning(
                    "Sample %d: modelled_positions has %d entries but tensor "
                    "has %d residues; skipping grounding for this sample",
                    b, len(mp), n_actual,
                )
                n_skipped_mismatch += 1
                continue

        # Build modelled_positions → tensor index mapping.
        # mp entries can be:
        #   - (chain, label_seq_id) tuples: chain-aware (multi-chain safe)
        #   - bare label_seq_id ints: legacy flat format (monomer-only)
        mp_to_idx = {}
        for i, pos in enumerate(mp):
            if isinstance(pos, (list, tuple)) and len(pos) == 2:
                mp_to_idx[(pos[0], pos[1])] = i
            elif pos is not None:
                mp_to_idx[pos] = i

        offsets = offset_mapping[b]  # [L_txt, 2]
        co = char_offsets[b]  # offset from chat template prefix

        for align in alignments:
            if align.get("type") == "fallback":
                continue

            text_chars = align.get("text_chars")
            residues = align.get("residues")
            if not text_chars or not residues:
                continue

            # Shift char span by chat template prefix offset
            char_start, char_end = text_chars[0] + co, text_chars[1] + co

            # Vectorized token matching: find all tokens overlapping [char_start, char_end)
            starts = offsets[:, 0]  # [L_txt]
            ends = offsets[:, 1]    # [L_txt]
            non_special = (starts != 0) | (ends != 0)
            overlaps = non_special & (ends > char_start) & (starts < char_end)
            tok_indices = overlaps.nonzero(as_tuple=True)[0].tolist()

            # Find residue tensor indices
            res_indices = []
            for res_entry in residues:
                if isinstance(res_entry, list) and len(res_entry) == 2:
                    chain, pos = res_entry
                    # Try chain-aware key first, fall back to flat
                    idx = mp_to_idx.get((chain, pos))
                    if idx is None:
                        idx = mp_to_idx.get(pos)
                else:
                    pos = res_entry
                    idx = mp_to_idx.get(pos)
                if idx is not None and idx < n_residues:
                    res_indices.append(idx)

            if tok_indices and res_indices:
                ti_t = torch.tensor(tok_indices, dtype=torch.long)
                ri_t = torch.tensor(res_indices, dtype=torch.long)
                # Broadcast set: grounding_mask[b, tok_indices, res_indices] = 1
                ti_grid, ri_grid = torch.meshgrid(ti_t, ri_t, indexing="ij")
                grounding_mask[b, ti_grid.reshape(-1), ri_grid.reshape(-1)] = 1.0
                token_has_mask[b, ti_t] = True

    stats = {
        "n_samples": batch_size,
        "n_skipped_mismatch": n_skipped_mismatch,
        "n_skipped_empty": n_skipped_empty,
    }
    return grounding_mask, token_has_mask, stats


def _to_position_key(entry) -> tuple | int | None:
    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
        return (str(entry[0]), int(entry[1]))
    if entry is None:
        return None
    return int(entry)


def _valid_ext_indices(idx_ext: Tensor, sample_idx: int) -> list[int]:
    idx = idx_ext[sample_idx].detach().cpu().tolist()
    return [int(x) for x in idx]


def _position_to_ext_map(
    positions: list,
    idx_ext: Tensor,
    sample_idx: int,
    add_tuple_flat_fallback: bool = False,
) -> dict:
    idx_values = _valid_ext_indices(idx_ext, sample_idx)
    out = {}
    for pos_i, pos in enumerate(positions or []):
        if pos_i >= len(idx_values):
            break
        ext_i = idx_values[pos_i]
        key = _to_position_key(pos)
        if key is not None and ext_i >= 0:
            out[key] = ext_i
            if add_tuple_flat_fallback and isinstance(key, tuple):
                out.setdefault(key[1], ext_i)
    return out


def _lookup_position(pos_map: dict, key):
    idx = pos_map.get(key)
    if idx is None and isinstance(key, tuple) and not any(isinstance(k, tuple) for k in pos_map):
        idx = pos_map.get(key[1])
    return idx


def build_extended_position_maps(
    binder_positions_list: list[list],
    target_positions_list: list[list],
    binder_idx_ext: Tensor,
    target_idx_ext: Tensor,
) -> tuple[list[dict[str, dict]], dict[str, int]]:
    """Build chain-aware residue-to-extended-index maps for compact concat.

    Returns per-sample maps with ``binder``, ``target``, and ``all`` entries.
    ``binder_positions_list`` and ``target_positions_list`` should contain
    ``(auth_chain, label_seq_id)`` tuples when chain IDs are available.
    """
    maps = []
    stats = {
        "n_samples": len(binder_positions_list),
        "n_binder_positions": 0,
        "n_target_positions": 0,
        "n_binder_mapped": 0,
        "n_target_mapped": 0,
    }
    for b in range(len(binder_positions_list)):
        binder_map = _position_to_ext_map(binder_positions_list[b], binder_idx_ext, b)
        target_map = _position_to_ext_map(target_positions_list[b], target_idx_ext, b)
        all_map = {}
        all_map.update(target_map)
        all_map.update(binder_map)
        maps.append({"binder": binder_map, "target": target_map, "all": all_map})
        stats["n_binder_positions"] += len(binder_positions_list[b] or [])
        stats["n_target_positions"] += len(target_positions_list[b] or [])
        stats["n_binder_mapped"] += len(set(v for v in binder_map.values() if isinstance(v, int)))
        stats["n_target_mapped"] += len(set(v for v in target_map.values() if isinstance(v, int)))
    return maps, stats


def _tokenize_with_offsets(
    texts: list[str],
    tokenizer,
    use_chat_template: bool,
) -> tuple[Tensor, list[int]]:
    tokenize_texts = texts
    char_offsets = [0] * len(texts)
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        tokenize_texts = []
        for b_idx, text in enumerate(texts):
            wrapped = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=False,
            )
            offset = wrapped.find(text)
            if offset < 0:
                logger.warning(
                    "Chat template modified text content; "
                    "grounding char offsets may be incorrect"
                )
                offset = 0
            char_offsets[b_idx] = offset
            tokenize_texts.append(wrapped)

    enc = tokenizer(
        tokenize_texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    return enc["offset_mapping"], char_offsets


def target_positions_from_batch(batch: dict) -> list[list[tuple]]:
    target_pdb_idx = batch.get("target_pdb_idx")
    target_chain_id = batch.get("target_chain_id")
    target_mask = batch.get("seq_target_mask")
    if target_pdb_idx is None:
        return [[] for _ in range(len(target_chain_id or []))]

    out = []
    B = target_pdb_idx.shape[0]
    for b in range(B):
        valid_n = int(target_mask[b].sum().item()) if target_mask is not None else target_pdb_idx.shape[1]
        chains_b = target_chain_id[b] if target_chain_id is not None and b < len(target_chain_id) else []
        idx_b = target_pdb_idx[b, :valid_n].detach().cpu().tolist()
        if len(chains_b) >= len(idx_b):
            chains_b = chains_b[:len(idx_b)]
            out.append([(chains_b[i], int(idx_b[i])) for i in range(len(idx_b))])
        else:
            out.append([int(x) for x in idx_b])
    return out


def build_extended_grounding_masks(
    texts: list[str],
    alignments_list: list[list[dict]],
    binder_modelled_positions_list: list[list],
    target_modelled_positions_list: list[list],
    tokenizer,
    binder_idx_ext: Tensor,
    target_idx_ext: Tensor,
    device: torch.device,
    use_chat_template: bool = False,
    facts: set[str] | None = None,
) -> tuple[Tensor, Tensor, dict[str, int]]:
    """Build text-token to extended-token masks for interface annotations."""
    facts = facts or INTERFACE_RESIDUE_FACTS
    batch_size = len(texts)
    n_ext = max(
        int(binder_idx_ext.max().item()) if binder_idx_ext.numel() else -1,
        int(target_idx_ext.max().item()) if target_idx_ext.numel() else -1,
    ) + 1
    n_ext = max(n_ext, 0)
    offset_mapping, char_offsets = _tokenize_with_offsets(texts, tokenizer, use_chat_template)
    n_txt = offset_mapping.shape[1]
    grounding_mask = torch.zeros(batch_size, n_txt, n_ext, device=device)
    token_has_mask = torch.zeros(batch_size, n_txt, dtype=torch.bool, device=device)

    position_maps, map_stats = build_extended_position_maps(
        binder_modelled_positions_list,
        target_modelled_positions_list,
        binder_idx_ext,
        target_idx_ext,
    )
    stats = {
        **map_stats,
        "n_annotations_total": 0,
        "n_annotations_used": 0,
        "n_residues_total": 0,
        "n_residues_resolved": 0,
        "n_residues_unresolved": 0,
    }

    for b in range(batch_size):
        offsets = offset_mapping[b]
        co = char_offsets[b]
        for align in alignments_list[b] or []:
            if align.get("type") == "fallback" or align.get("fact") not in facts:
                continue
            text_chars = align.get("text_chars")
            residues = align.get("residues")
            if not text_chars or not residues:
                continue
            stats["n_annotations_total"] += 1
            char_start, char_end = text_chars[0] + co, text_chars[1] + co
            starts = offsets[:, 0]
            ends = offsets[:, 1]
            non_special = (starts != 0) | (ends != 0)
            tok_indices = (non_special & (ends > char_start) & (starts < char_end)).nonzero(as_tuple=True)[0]
            if tok_indices.numel() == 0:
                continue

            ext_indices = []
            for res_entry in residues:
                stats["n_residues_total"] += 1
                key = _to_position_key(res_entry)
                idx = _lookup_position(position_maps[b]["all"], key)
                if idx is not None:
                    ext_indices.append(idx)
                    stats["n_residues_resolved"] += 1
                else:
                    stats["n_residues_unresolved"] += 1
            if ext_indices:
                ri_t = torch.tensor(ext_indices, dtype=torch.long, device=device)
                ti_t = tok_indices.to(device=device, dtype=torch.long)
                ti_grid, ri_grid = torch.meshgrid(ti_t, ri_t, indexing="ij")
                grounding_mask[b, ti_grid.reshape(-1), ri_grid.reshape(-1)] = 1.0
                token_has_mask[b, ti_t] = True
                stats["n_annotations_used"] += 1

    return grounding_mask, token_has_mask, stats


def build_pair_grounding_masks(
    alignments_list: list[list[dict]],
    binder_modelled_positions_list: list[list],
    target_modelled_positions_list: list[list],
    binder_idx_ext: Tensor,
    target_idx_ext: Tensor,
    device: torch.device,
    facts: set[str] | None = None,
) -> tuple[Tensor, Tensor, dict[str, int]]:
    """Build symmetric extended pair labels from interface ``pairs`` fields."""
    facts = facts or INTERFACE_PAIR_FACTS
    batch_size = len(alignments_list)
    n_ext = max(
        int(binder_idx_ext.max().item()) if binder_idx_ext.numel() else -1,
        int(target_idx_ext.max().item()) if target_idx_ext.numel() else -1,
    ) + 1
    n_ext = max(n_ext, 0)
    pair_mask = torch.zeros(batch_size, n_ext, n_ext, device=device)
    pair_has_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
    position_maps, map_stats = build_extended_position_maps(
        binder_modelled_positions_list,
        target_modelled_positions_list,
        binder_idx_ext,
        target_idx_ext,
    )
    stats = {
        **map_stats,
        "n_with_pair_annotations": 0,
        "n_pairs_total": 0,
        "n_pairs_resolved": 0,
        "n_pairs_unresolved_binder": 0,
        "n_pairs_unresolved_target": 0,
        "n_interface_hbond_pairs": 0,
        "n_interface_salt_pairs": 0,
        "n_interface_hydrophobic_pairs": 0,
    }

    fact_to_stat = {
        "interface_hbonds": "n_interface_hbond_pairs",
        "interface_salt_bridges": "n_interface_salt_pairs",
        "interface_hydrophobic_contacts": "n_interface_hydrophobic_pairs",
    }

    for b, alignments in enumerate(alignments_list):
        sample_had_pairs = False
        for align in alignments or []:
            fact = align.get("fact")
            pairs = align.get("pairs") or []
            if fact not in facts or not pairs:
                continue
            sample_had_pairs = True
            for pair in pairs:
                if len(pair) != 2:
                    continue
                stats["n_pairs_total"] += 1
                if fact in fact_to_stat:
                    stats[fact_to_stat[fact]] += 1
                a_key = _to_position_key(pair[0])
                b_key = _to_position_key(pair[1])
                binder_a = _lookup_position(position_maps[b]["binder"], a_key)
                target_b = _lookup_position(position_maps[b]["target"], b_key)
                binder_b = _lookup_position(position_maps[b]["binder"], b_key)
                target_a = _lookup_position(position_maps[b]["target"], a_key)
                if binder_a is not None and target_b is not None:
                    i, j = binder_a, target_b
                elif binder_b is not None and target_a is not None:
                    i, j = binder_b, target_a
                else:
                    if binder_a is None and binder_b is None:
                        stats["n_pairs_unresolved_binder"] += 1
                    if target_a is None and target_b is None:
                        stats["n_pairs_unresolved_target"] += 1
                    continue
                pair_mask[b, i, j] = 1.0
                pair_mask[b, j, i] = 1.0
                pair_has_mask[b] = True
                stats["n_pairs_resolved"] += 1
        if sample_had_pairs:
            stats["n_with_pair_annotations"] += 1

    return pair_mask, pair_has_mask, stats


def compute_token_loss(
    attn: Tensor,
    mask: Tensor,
    token_has_mask: Tensor,
    res_mask: Tensor,
) -> Tensor:
    """DEPRECATED: use AlignmentHead instead.

    Token-level grounding loss: attention concentration inside annotated region.

    L_token = (1/N) * sum_i [ (1 - sum_{u in B_i} A_{i,u} / sum_u A_{i,u})^2 ]

    Args:
        attn: [B, N_txt, N_res] head-averaged cross-attention weights.
        mask: [B, N_txt, N_res] binary grounding mask.
        token_has_mask: [B, N_txt] bool, True for tokens with annotations.
        res_mask: [B, N_res] bool, True for valid (non-padding) residues.

    Returns:
        Per-sample loss [B].
    """
    # Mask out padded residues
    attn_valid = attn * res_mask[:, None, :]

    in_region = (attn_valid * mask).sum(dim=-1)  # [B, N_txt]
    total = attn_valid.sum(dim=-1).clamp(min=1e-8)  # [B, N_txt]
    ratio = in_region / total  # [B, N_txt]
    per_token = (1.0 - ratio) ** 2  # [B, N_txt]

    n_annotated = token_has_mask.float().sum(dim=-1).clamp(min=1.0)  # [B]
    loss = (per_token * token_has_mask.float()).sum(dim=-1) / n_annotated
    return loss  # [B]


def compute_pixel_loss(
    attn: Tensor,
    mask: Tensor,
    token_has_mask: Tensor,
    res_mask: Tensor,
) -> Tensor:
    """DEPRECATED: use AlignmentHead instead.

    Pixel-level grounding loss: per-position binary cross-entropy.

    L_pixel = -(1/(N_tok * N_res)) sum_i sum_u [ M*log(A) + (1-M)*log(1-A) ]

    Only computed over annotated tokens and valid residues.

    Args:
        attn: [B, N_txt, N_res] head-averaged cross-attention weights.
        mask: [B, N_txt, N_res] binary grounding mask.
        token_has_mask: [B, N_txt] bool, True for tokens with annotations.
        res_mask: [B, N_res] bool, True for valid (non-padding) residues.

    Returns:
        Per-sample loss [B].
    """
    eps = 1e-7
    attn_c = attn.clamp(eps, 1.0 - eps)
    bce = -(mask * torch.log(attn_c) + (1.0 - mask) * torch.log(1.0 - attn_c))

    # Only count annotated-token x valid-residue pairs
    valid = token_has_mask[:, :, None].float() * res_mask[:, None, :].float()
    n_valid = valid.sum(dim=(-1, -2)).clamp(min=1.0)
    loss = (bce * valid).sum(dim=(-1, -2)) / n_valid
    return loss  # [B]


def compute_grounding_loss(
    attn_maps: list[Tensor],
    grounding_mask: Tensor,
    token_has_mask: Tensor,
    res_mask: Tensor,
    lambda_token: float = 1.0,
    gamma_pixel: float = 0.1,
    layer_weights: GroundingLayerWeights | None = None,
) -> Tensor:
    """DEPRECATED: use AlignmentHead instead.

    Combined grounding loss across supervised cross-attention layers.

    When ``layer_weights`` is None, uses uniform averaging:
        L = (1/M) * sum_m [ lambda * L_token^(m) + gamma * L_pixel^(m) ]

    When ``layer_weights`` is provided, uses learnable softmax weights:
        L = sum_m [ w_m * (lambda * L_token^(m) + gamma * L_pixel^(m)) ]

    Args:
        attn_maps: List of [B, N_res, N_txt] attention weight tensors (one per layer).
        grounding_mask: [B, N_txt, N_res] binary grounding mask.
        token_has_mask: [B, N_txt] bool.
        res_mask: [B, N_res] bool.
        lambda_token: Weight for token loss.
        gamma_pixel: Weight for pixel loss.
        layer_weights: Optional learnable layer weighting module.

    Returns:
        Per-sample loss [B].
    """
    warnings.warn(
        "compute_grounding_loss is deprecated (requires need_weights=True, "
        "disables flash attention). Use AlignmentHead instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    batch_size = grounding_mask.shape[0]
    device = grounding_mask.device
    n_layers = len(attn_maps)

    if layer_weights is not None:
        w = layer_weights()  # [n_layers], softmax-normalized
        if len(w) != n_layers:
            logger.warning(
                f"layer_weights has {len(w)} entries but got {n_layers} attn maps; "
                "falling back to uniform weighting"
            )
            w = None
    else:
        w = None

    # Reconcile text dimension once before the loop (avoid mutating
    # grounding_mask/token_has_mask on each iteration).
    n_txt_attn = attn_maps[0].shape[2]
    n_txt_mask = grounding_mask.shape[1]
    if n_txt_attn != n_txt_mask:
        min_txt = min(n_txt_attn, n_txt_mask)
        grounding_mask = grounding_mask[:, :min_txt, :]
        token_has_mask = token_has_mask[:, :min_txt]

    total = torch.zeros(batch_size, device=device)

    for m, attn_map in enumerate(attn_maps):
        # attn_map from nn.MultiheadAttention: [B, N_res, N_txt]
        # Truncate text dim to match reconciled mask if needed
        if attn_map.shape[2] != grounding_mask.shape[1]:
            attn_map = attn_map[:, :, :grounding_mask.shape[1]]
        A = attn_map.transpose(1, 2)

        l_token = compute_token_loss(A, grounding_mask, token_has_mask, res_mask)
        l_pixel = compute_pixel_loss(A, grounding_mask, token_has_mask, res_mask)

        layer_loss = lambda_token * l_token + gamma_pixel * l_pixel

        if w is not None:
            total = total + w[m] * layer_loss
        else:
            total = total + layer_loss

    # Uniform average when no learnable weights
    if w is None:
        total = total / max(n_layers, 1)

    return total
