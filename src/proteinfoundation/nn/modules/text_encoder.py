from typing import Optional
import os

import torch
import torch.nn as nn
from loguru import logger

from proteinfoundation.nn.modules.text_projector import build_text_projector


class TextEncoder(nn.Module):
    """Frozen text encoder(s) + trainable projector(s) for text-conditioned generation.

    Manages:
      - Pooled path: encoder → projector → [B, dim_cond]
      - Token path: encoder → projector → [B, L, token_dim]
      - CFG dropout (zeroing during training)
      - Unique-text deduplication for efficiency

    Frozen encoders are stored as plain Python attributes (NOT registered
    submodules) to prevent EMA from maintaining shadow copies of ~16GB
    frozen weights and to keep them out of state_dict() automatically.

    forward() returns (pooled_cond, token_cond, token_mask) tensors.
    Batch mutation happens in proteina.py.
    """

    def __init__(self, cfg, dim_cond: int, token_dim: int):
        """
        Args:
            cfg: text_cond config (OmegaConf).
            dim_cond: Denoiser conditioning dimension (e.g., 256).
            token_dim: Denoiser token dimension (e.g., 768).
        """
        super().__init__()
        self.p_uncond = float(getattr(cfg, "p_uncond", 0.0))
        self.p_uncond_t_schedule = getattr(cfg, "p_uncond_t_schedule", None)
        proj_hidden = getattr(cfg, "projector_hidden_dim", None)
        # Move the frozen encoders back to CPU after encoding. Frees roughly
        # 32GB of GPU memory at the cost of a transfer per batch.
        self.offload_after_encode = bool(getattr(cfg, "offload_text_encoders", False))

        # Parse config
        pooled_path = getattr(cfg, "pooled_path", None)
        token_path = getattr(cfg, "token_path", None)

        self.use_pooled = pooled_path is not None and getattr(pooled_path, "enabled", False)
        self.use_tokens = token_path is not None and getattr(token_path, "enabled", False)

        # --- Pooled encoder (plain attribute, NOT a submodule) ---
        # Use object.__setattr__ to bypass nn.Module.__setattr__, which
        # would register any nn.Module as a submodule (appearing in
        # state_dict(), parameters(), and EMA shadow copies).
        object.__setattr__(self, "_pooled_model", None)
        object.__setattr__(self, "_pooled_tokenizer", None)
        object.__setattr__(self, "_pooled_backend", None)
        object.__setattr__(self, "_pooled_instruct_task", None)
        object.__setattr__(self, "_pooled_kind", None)
        self.pooled_proj = None  # registered submodule (trainable)

        if self.use_pooled:
            self._init_pooled_path(pooled_path, cfg, dim_cond, proj_hidden)

        # --- Token encoder (plain attribute, NOT a submodule) ---
        object.__setattr__(self, "_tokens_model", None)
        object.__setattr__(self, "_tokens_tokenizer", None)
        object.__setattr__(self, "_tokens_use_chat_template", False)
        object.__setattr__(self, "_tokens_model_class", "auto")
        object.__setattr__(self, "_token_layer_mode", "last")
        object.__setattr__(self, "_token_layer_index", None)
        object.__setattr__(self, "_token_final_norm", None)
        self.tokens_proj = None  # registered submodule (trainable)

        if self.use_tokens:
            self._init_token_path(token_path, cfg, token_dim, proj_hidden)

    def _init_pooled_path(self, pooled_path, cfg, dim_cond, proj_hidden):
        """Initialize the pooled text encoder and projector."""
        pooled_model_cfg = pooled_path.model
        pooled_name = pooled_model_cfg.model_name
        object.__setattr__(self, "_pooled_backend", getattr(pooled_model_cfg, "backend", "transformers"))
        object.__setattr__(self, "_pooled_instruct_task", getattr(pooled_model_cfg, "instruct_task", None))
        object.__setattr__(self, "_pooled_kind", getattr(pooled_model_cfg, "kind", "embedding"))

        if self._pooled_backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(pooled_name, device="cpu", trust_remote_code=True)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            object.__setattr__(self, "_pooled_model", model)
            object.__setattr__(self, "_pooled_tokenizer", None)
            pooled_dim = model[0].get_word_embedding_dimension()
        else:
            from transformers import AutoModel, AutoTokenizer

            object.__setattr__(self, "_pooled_tokenizer", AutoTokenizer.from_pretrained(pooled_name))
            model = AutoModel.from_pretrained(pooled_name)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            object.__setattr__(self, "_pooled_model", model)

            pooled_dim = getattr(model.config, "hidden_size", None)
            if pooled_dim is None:
                pooled_dim = getattr(pooled_model_cfg, "embed_dim", None)
                if pooled_dim is None:
                    raise ValueError(
                        "text_cond.pooled_path.model must have hidden_size in config "
                        "or embed_dim specified"
                    )

        # Create projector (registered submodule, trainable)
        pooled_proj_cfg = getattr(cfg, "pooled_projector", None)
        pooled_arch = getattr(pooled_proj_cfg, "arch", "mlp") if pooled_proj_cfg else "mlp"
        pooled_proj_kwargs = {}
        if pooled_arch == "deep_mlp":
            pooled_proj_kwargs["num_layers"] = getattr(pooled_proj_cfg, "num_layers", 3)
        elif pooled_arch == "attention_pool":
            pooled_proj_kwargs["num_queries"] = getattr(pooled_proj_cfg, "num_queries", 8)
            pooled_proj_kwargs["num_heads"] = getattr(pooled_proj_cfg, "num_heads", 8)

        self.pooled_proj = build_text_projector(
            arch=pooled_arch,
            d_in=pooled_dim,
            d_out=dim_cond,
            d_hidden=proj_hidden,
            **pooled_proj_kwargs,
        )
        logger.info(
            f"TextEncoder pooled path: {pooled_name} ({self._pooled_backend}) "
            f"dim={pooled_dim} → proj({pooled_arch}) → {dim_cond}"
        )

    def _init_token_path(self, token_path, cfg, token_dim, proj_hidden):
        """Initialize the token-level text encoder and projector."""
        from transformers import AutoModel, AutoTokenizer

        token_model_cfg = token_path.model
        tokens_name = token_model_cfg.model_name
        model_class = getattr(token_model_cfg, "model_class", "auto")
        object.__setattr__(self, "_tokens_use_chat_template", getattr(token_model_cfg, "use_chat_template", False))
        object.__setattr__(self, "_tokens_model_class", model_class)

        # Check if we can reuse the pooled model (same LM for both paths)
        reuse_lm = False
        if self.use_pooled and self._pooled_model is not None:
            pooled_path_cfg = getattr(cfg, "pooled_path", None)
            if pooled_path_cfg is not None:
                pooled_model_cfg = pooled_path_cfg.model
                pooled_kind = getattr(pooled_model_cfg, "kind", "embedding")
                if pooled_kind == "lm" and tokens_name == pooled_model_cfg.model_name:
                    reuse_lm = True

        if reuse_lm:
            object.__setattr__(self, "_tokens_model", self._pooled_model)
            object.__setattr__(self, "_tokens_tokenizer", self._pooled_tokenizer)
            tokens_dim = self._pooled_model.config.hidden_size
            logger.info(f"TextEncoder token path: reusing pooled LM ({tokens_name})")
        elif model_class == "t5_encoder":
            from transformers import AutoTokenizer, T5EncoderModel

            tokenizer = AutoTokenizer.from_pretrained(tokens_name)
            model = T5EncoderModel.from_pretrained(tokens_name)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

            object.__setattr__(self, "_tokens_model", model)
            object.__setattr__(self, "_tokens_tokenizer", tokenizer)
            tokens_dim = model.config.d_model
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokens_name)
            model = AutoModel.from_pretrained(tokens_name)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False

            # For multimodal models, extract the text backbone
            if hasattr(model, "language_model"):
                model = model.language_model
            elif hasattr(model, "text_model"):
                model = model.text_model

            if os.environ.get("COMPILE_TEXT_ENCODERS", "0") == "1":
                model = torch.compile(model)
            object.__setattr__(self, "_tokens_model", model)
            object.__setattr__(self, "_tokens_tokenizer", tokenizer)

            tokens_dim = getattr(model.config, "hidden_size", None)
            if tokens_dim is None:
                text_cfg = getattr(model.config, "text_config", None)
                if text_cfg is not None:
                    tokens_dim = getattr(text_cfg, "hidden_size", None)
            if tokens_dim is None:
                tokens_dim = getattr(token_model_cfg, "embed_dim", None)
                if tokens_dim is None:
                    raise ValueError(
                        "text_cond.token_path.model must have hidden_size in config "
                        "or embed_dim specified"
                    )

        # Layer selection mode
        object.__setattr__(self, "_token_layer_mode", getattr(token_path, "layer_mode", "last"))
        object.__setattr__(self, "_token_layer_index", getattr(token_path, "layer_index", None))
        if self._token_layer_mode == "index" and self._token_layer_index is None:
            raise ValueError("token_path.layer_index required when layer_mode='index'")

        # Cache reference to final norm for intermediate layer normalization.
        # When using average/index modes, hidden states from non-final layers
        # need normalization because raw hidden states have different scales.
        # This is a reference to a child of the frozen encoder — it follows
        # the encoder's device when the encoder is moved via .to(device).
        if self._token_layer_mode != "last":
            tok_model = self._tokens_model
            if hasattr(tok_model, "encoder") and hasattr(tok_model.encoder, "final_layer_norm"):
                # T5EncoderModel: encoder.final_layer_norm
                object.__setattr__(self, "_token_final_norm", tok_model.encoder.final_layer_norm)
            elif hasattr(tok_model, "model") and hasattr(tok_model.model, "norm"):
                # Decoder-only LMs (Qwen, Llama, etc.): model.norm
                object.__setattr__(self, "_token_final_norm", tok_model.model.norm)
            elif hasattr(tok_model, "norm"):
                object.__setattr__(self, "_token_final_norm", tok_model.norm)
            else:
                raise ValueError("Cannot find final norm layer on text token model")

        # Create projector (registered submodule, trainable)
        token_proj_cfg = getattr(cfg, "token_projector", None)
        token_arch = getattr(token_proj_cfg, "arch", "mlp") if token_proj_cfg else "mlp"
        if token_arch == "attention_pool":
            raise ValueError("attention_pool is not supported for the token projector path")
        token_proj_kwargs = {}
        if token_arch == "deep_mlp":
            token_proj_kwargs["num_layers"] = getattr(token_proj_cfg, "num_layers", 3)

        self.tokens_proj = build_text_projector(
            arch=token_arch,
            d_in=tokens_dim,
            d_out=token_dim,
            d_hidden=proj_hidden,
            **token_proj_kwargs,
        )
        logger.info(
            f"TextEncoder token path: {tokens_name} "
            f"dim={tokens_dim} layer_mode={self._token_layer_mode} "
            f"→ proj({token_arch}) → {token_dim}"
        )

    def _get_p_uncond(self, t: torch.Tensor | None = None) -> float | torch.Tensor:
        """Compute per-sample CFG dropout probability, optionally timestep-dependent.

        When p_uncond_t_schedule is configured, the dropout probability varies
        with the flow-matching timestep t (0=noise, 1=clean):
          - "linear_decay": p(t) = p_base * (1 - t). More dropout at noisy
            timesteps (where text is less useful), less near clean data.
          - "linear_increase": p(t) = p_base * t. More dropout near clean
            data (forces the model to rely on structure, not text shortcuts).

        Args:
            t: Per-sample timestep [B] or None.

        Returns:
            Scalar p_uncond (when no schedule or t is None), or per-sample tensor [B].
        """
        if self.p_uncond_t_schedule is None or t is None:
            return self.p_uncond

        mode = self.p_uncond_t_schedule
        if mode == "linear_decay":
            return self.p_uncond * (1.0 - t)
        elif mode == "linear_increase":
            return self.p_uncond * t
        else:
            return self.p_uncond

    def forward(
        self,
        texts: list[str],
        device: torch.device,
        training: bool = False,
        t: torch.Tensor | None = None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Encode texts into pooled and/or token-level embeddings.

        Args:
            texts: List of text strings, length = batch size.
            device: Target device for output tensors.
            training: Whether in training mode (for CFG dropout).
            t: Per-sample timestep tensor [B], used for timestep-dependent
               CFG dropout when p_uncond_t_schedule is configured.

        Returns:
            (pooled_cond, token_cond, token_mask, cfg_keep_mask):
              - pooled_cond: [B, dim_cond] or None
              - token_cond:  [B, L, token_dim] or None
              - token_mask:  [B, L] bool or None
              - cfg_keep_mask: [B] bool or None (True = text kept, False = dropped for CFG)
        """
        if isinstance(texts, list) and len(texts) > 0 and isinstance(texts[0], tuple):
            texts = [t[0] if isinstance(t, tuple) and len(t) == 1 else t for t in texts]

        if not isinstance(texts, (list, tuple)):
            raise ValueError("Expected texts to be a list of strings")

        # Deduplicate texts for efficiency
        unique_texts_list = []
        text_to_idx = {}
        original_to_unique = []
        for text in texts:
            if text not in text_to_idx:
                text_to_idx[text] = len(unique_texts_list)
                unique_texts_list.append(text)
            original_to_unique.append(text_to_idx[text])

        pooled_states = None
        token_states = None
        token_mask = None

        # --- Encode under no_grad (encoders are frozen) ---
        with torch.no_grad():
            if self.use_pooled and self._pooled_model is not None:
                pooled_states = self._encode_pooled(unique_texts_list, device)

            if self.use_tokens and self._tokens_model is not None:
                token_states, token_mask = self._encode_tokens(unique_texts_list, device)

        # --- Apply trainable projectors (these get gradients) ---
        pooled_cond = None
        token_cond = None

        if pooled_states is not None and self.pooled_proj is not None:
            proj_dtype = next(self.pooled_proj.parameters()).dtype
            if pooled_states.dtype != proj_dtype:
                pooled_states = pooled_states.to(proj_dtype)
            pooled_cond = self.pooled_proj(pooled_states)  # [num_unique, dim_cond]

        if token_states is not None and self.tokens_proj is not None:
            proj_dtype = next(self.tokens_proj.parameters()).dtype
            if token_states.dtype != proj_dtype:
                token_states = token_states.to(proj_dtype)
            token_cond = self.tokens_proj(token_states)  # [num_unique, L, token_dim]

        # --- Map back to original batch order ---
        idx = torch.tensor(original_to_unique, device=device)
        if pooled_cond is not None:
            pooled_cond = pooled_cond[idx]  # [B, dim_cond]
        if token_cond is not None:
            token_cond = token_cond[idx]  # [B, L, token_dim]
            if token_mask is not None:
                token_mask = token_mask[idx]  # [B, L]

        # --- Zero embeddings for empty text strings ---
        # Samples with empty text (e.g. from binder-chain filtering or
        # missing captions) should produce exactly-zero conditioning,
        # matching the CFG unconditional signal.
        empty_mask = torch.tensor(
            [len(t.strip()) == 0 for t in texts], device=device, dtype=torch.bool
        )
        if empty_mask.any():
            keep_text = (~empty_mask).float()
            if pooled_cond is not None:
                pooled_cond = pooled_cond * keep_text[:, None]
            if token_cond is not None:
                token_cond = token_cond * keep_text[:, None, None]

        # --- CFG dropout: zero embeddings for dropped samples ---
        # Only zero the embeddings, NOT the mask. Zeroed embeddings with
        # valid mask positions produce zero attention output (correct
        # unconditional signal). Zeroing the mask would cause all keys
        # to be ignored → softmax(-inf) → NaN in cross-attention.
        cfg_keep_mask = None
        if self.p_uncond > 0.0 and training:
            b = len(texts)
            p_drop = self._get_p_uncond(t)
            keep = (torch.rand(b, device=device) >= p_drop)  # True = keep
            cfg_keep_mask = keep

            if pooled_cond is not None:
                pooled_cond = pooled_cond * keep[:, None]
            if token_cond is not None:
                token_cond = token_cond * keep[:, None, None]

        return pooled_cond, token_cond, token_mask, cfg_keep_mask

    def _encode_pooled(self, texts: list[str], device: torch.device) -> torch.Tensor:
        """Encode texts into pooled embeddings using the frozen encoder."""
        encode_texts = texts
        if self._pooled_instruct_task:
            encode_texts = [
                f"Instruct: {self._pooled_instruct_task}\nQuery: {t}"
                for t in texts
            ]

        if self._pooled_backend == "sentence_transformers":
            self._pooled_model.to(device)
            emb = self._pooled_model.encode(
                encode_texts,
                batch_size=len(encode_texts),
                convert_to_numpy=False,
                normalize_embeddings=True,
            )
            if isinstance(emb, list):
                emb = torch.stack(emb)
            result = emb.to(device)
        else:
            enc = self._pooled_tokenizer(
                encode_texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            self._pooled_model.to(device)
            out = self._pooled_model(**enc)

            if self._pooled_kind == "lm":
                mask = enc["attention_mask"].unsqueeze(-1).float()
                result = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                result = out.last_hidden_state.mean(dim=1)

        if self.offload_after_encode and self._pooled_model is not None:
            self._pooled_model.to("cpu")
            torch.cuda.empty_cache()
        return result

    def _encode_tokens(
        self, texts: list[str], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode texts into token-level hidden states using the frozen encoder."""
        token_texts = texts
        if self._tokens_use_chat_template and hasattr(self._tokens_tokenizer, "apply_chat_template"):
            token_texts = [
                self._tokens_tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                for t in texts
            ]

        enc_tok = self._tokens_tokenizer(
            token_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc_tok = {k: v.to(device) for k, v in enc_tok.items()}

        need_hidden = self._token_layer_mode != "last"
        self._tokens_model.to(device)
        out_tok = self._tokens_model(**enc_tok, output_hidden_states=need_hidden)

        if self._token_layer_mode == "last":
            token_states = out_tok.last_hidden_state
        elif self._token_layer_mode == "index":
            token_states = out_tok.hidden_states[self._token_layer_index]
            token_states = self._token_final_norm(token_states)
        elif self._token_layer_mode == "average":
            # T5 includes post-norm output as hidden_states[-1]; exclude it
            # to avoid double-normalization. Decoder-only LMs store pre-norm
            # states throughout, so all layers need the final norm applied.
            end = -1 if self._tokens_model_class == "t5_encoder" else None
            all_layers = torch.stack(out_tok.hidden_states[1:end], dim=0)
            all_layers = self._token_final_norm(all_layers)
            token_states = all_layers.mean(dim=0)
        else:
            raise ValueError(f"Unknown token_layer_mode: {self._token_layer_mode}")

        if self.offload_after_encode and self._tokens_model is not None:
            self._tokens_model.to("cpu")
            torch.cuda.empty_cache()

        token_mask = enc_tok["attention_mask"].bool()  # [B, L]
        return token_states, token_mask
