"""Pre-compute text embeddings for length predictor training.

Encodes protein descriptions with both frozen Qwen encoders and saves
raw (pre-projection) outputs as .pt files for offline training.

Pooled path: Qwen3-Embedding-8B via SentenceTransformers (with instruct prefix)
Token path:  Qwen3-8B LM, layer-averaged hidden states with final norm

Usage:
    python scripts/precompute_text_embeddings.py \
        --csv_dir $DATA_PATH/length_predictor/v3_subset \
        --out_dir $DATA_PATH/length_predictor/v3_subset_embeddings \
        --batch_size 8

Processes train.csv, val.csv, test.csv from csv_dir and writes per-split
subdirectories with pooled.pt, tokens.pt, token_masks.pt, lengths.pt.
"""

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm


POOLED_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
TOKEN_MODEL_NAME = "Qwen/Qwen3-8B"
INSTRUCT_TASK = (
    "Given a description of a protein, embed it based on semantic meaning only. "
    "Ignore surface form variation such as casing, punctuation, paraphrasing or rewording."
)


def encode_pooled(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode texts into pooled embeddings using SentenceTransformers."""
    from sentence_transformers import SentenceTransformer

    print(f"Loading pooled encoder: {model_name}")
    model = SentenceTransformer(model_name, device=str(device), trust_remote_code=True)
    model.eval()

    # Apply instruct prefix
    prefixed = [f"Instruct: {INSTRUCT_TASK}\nQuery: {t}" for t in texts]

    all_embs = []
    for i in tqdm(range(0, len(prefixed), batch_size), desc="Pooled encoding"):
        batch = prefixed[i : i + batch_size]
        with torch.no_grad():
            emb = model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=False,
                normalize_embeddings=True,
            )
            if isinstance(emb, list):
                emb = torch.stack(emb)
            all_embs.append(emb.cpu().float())

    # Unload
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return torch.cat(all_embs, dim=0)


def encode_tokens(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: torch.device,
    max_length: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode texts into layer-averaged token embeddings.

    Matches TextEncoder._encode_tokens with layer_mode='average':
    - Apply chat template
    - Get all hidden states
    - Average layers 1..N (skip embedding layer 0)
    - Apply final norm
    """
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading token encoder: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    # Get final norm reference (matches text_encoder.py logic)
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        final_norm = model.model.norm
    elif hasattr(model, "norm"):
        final_norm = model.norm
    else:
        raise RuntimeError("Cannot find final norm layer on token model")

    # Apply chat template
    templated = []
    for t in texts:
        msg = tokenizer.apply_chat_template(
            [{"role": "user", "content": t}],
            tokenize=False,
            add_generation_prompt=False,
        )
        templated.append(msg)

    all_tokens_list = []
    all_masks_list = []
    max_seq_len = 0

    for i in tqdm(range(0, len(templated), batch_size), desc="Token encoding"):
        batch = templated[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)

            # Layer-averaged hidden states (skip layer 0 = embedding)
            # Must stay inside no_grad — final_norm has parameters
            n_layers = len(out.hidden_states) - 1
            avg = torch.zeros_like(out.hidden_states[1])
            for layer_state in out.hidden_states[1:]:
                avg += final_norm(layer_state)
            token_states = avg / n_layers  # [B, T, D]

            del out, avg

        mask = enc["attention_mask"].bool()  # [B, T]

        all_tokens_list.append(token_states.cpu().half())
        all_masks_list.append(mask.cpu())
        max_seq_len = max(max_seq_len, token_states.shape[1])

    # Unload
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Pre-allocate output and copy chunks to avoid 2x peak memory from torch.cat
    N = sum(t.shape[0] for t in all_tokens_list)
    D = all_tokens_list[0].shape[-1]
    tokens_out = torch.zeros(N, max_seq_len, D, dtype=torch.float16)
    masks_out = torch.zeros(N, max_seq_len, dtype=torch.bool)

    offset = 0
    for i in range(len(all_tokens_list)):
        chunk_t = all_tokens_list[i]
        chunk_m = all_masks_list[i]
        B, T, _ = chunk_t.shape
        tokens_out[offset:offset + B, :T] = chunk_t
        masks_out[offset:offset + B, :T] = chunk_m
        # Free chunk immediately
        all_tokens_list[i] = None
        all_masks_list[i] = None
        offset += B

    del all_tokens_list, all_masks_list
    gc.collect()

    return tokens_out, masks_out


def process_split(
    csv_path: Path,
    out_dir: Path,
    batch_size: int,
    device: torch.device,
    pooled_model: str,
    token_model: str,
    shard_idx: int | None = None,
    num_shards: int | None = None,
):
    """Process a single CSV split into .pt embedding files.

    When shard_idx/num_shards are set, only processes rows
    shard_idx::num_shards and saves to out_dir/shard_{shard_idx:03d}/.
    """
    df = pd.read_csv(csv_path, dtype={"accession": str, "description": str, "length": int})

    if shard_idx is not None and num_shards is not None:
        df = df.iloc[shard_idx::num_shards].reset_index(drop=True)
        out_dir = out_dir / f"shard_{shard_idx:03d}"

    texts = df["description"].tolist()
    lengths = torch.tensor(df["length"].values, dtype=torch.long) - 1  # 0-indexed

    shard_info = f" (shard {shard_idx}/{num_shards})" if shard_idx is not None else ""
    print(f"\n{'='*60}")
    print(f"Processing {csv_path.name}{shard_info}: {len(texts)} samples")
    print(f"{'='*60}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Pooled embeddings
    pooled = encode_pooled(texts, pooled_model, batch_size, device)
    torch.save(pooled, out_dir / "pooled.pt")
    print(f"  Saved pooled.pt: {pooled.shape}")
    del pooled
    gc.collect()
    torch.cuda.empty_cache()

    # Token embeddings (use smaller batch size — layer averaging is memory-intensive)
    token_batch_size = max(1, batch_size // 2)
    tokens, masks = encode_tokens(texts, token_model, token_batch_size, device)
    torch.save(tokens, out_dir / "tokens.pt")
    torch.save(masks, out_dir / "token_masks.pt")
    print(f"  Saved tokens.pt: {tokens.shape}")
    print(f"  Saved token_masks.pt: {masks.shape}")
    del tokens, masks
    gc.collect()
    torch.cuda.empty_cache()

    # Lengths
    torch.save(lengths, out_dir / "lengths.pt")
    print(f"  Saved lengths.pt: {lengths.shape}")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute text embeddings")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Directory containing train.csv, val.csv, test.csv")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for .pt files")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"],
                        help="Splits to process")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--pooled_model", type=str, default=POOLED_MODEL_NAME)
    parser.add_argument("--token_model", type=str, default=TOKEN_MODEL_NAME)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shard_idx", type=int, default=None,
                        help="Shard index (0-based). Processes rows shard_idx::num_shards.")
    parser.add_argument("--num_shards", type=int, default=None,
                        help="Total number of shards.")
    args = parser.parse_args()

    if (args.shard_idx is None) != (args.num_shards is None):
        parser.error("--shard_idx and --num_shards must be specified together")

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    device = torch.device(args.device)

    for split in args.splits:
        csv_path = csv_dir / f"{split}.csv"
        if not csv_path.exists():
            print(f"Skipping {split}: {csv_path} not found")
            continue
        process_split(
            csv_path=csv_path,
            out_dir=out_dir / split,
            batch_size=args.batch_size,
            device=device,
            pooled_model=args.pooled_model,
            token_model=args.token_model,
            shard_idx=args.shard_idx,
            num_shards=args.num_shards,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
