import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# dbt Python model entrypoint
def model(dbt, session):
    """
    - Reads raw_documents (id, text) via dbt.ref('raw_documents') or falls back to direct table
    - Computes embeddings with HF Transformers (LiquidAI/LFM2-ColBERT-350M) using mean pooling
      Note: sentence-transformers fails with 'MaxSim' SimilarityFunction for this model;
            we therefore load via transformers and pool token embeddings to a single vector.
    - Returns a DataFrame with columns: id, text, embedding (list[float])
    - Materializes as a DuckDB table named final_documents
    """
    dbt.config(materialized="table", alias="final_documents")

    # 1) Load source data (prefer dbt.ref; fallback to direct table)
    df = None
    try:
        rel = dbt.ref("raw_documents")
        try:
            if hasattr(rel, "to_df"):
                df = rel.to_df()
            elif hasattr(rel, "to_pandas"):
                df = rel.to_pandas()
            else:
                df = session.sql(f"SELECT id, text FROM {rel}").df()
        except Exception:
            df = session.sql(f"SELECT id, text FROM {rel}").df()
    except Exception:
        df = session.sql("SELECT id, text FROM raw_documents").df()

    if df is None or df.empty:
        raise ValueError("raw_documents から読み込めるデータが見つかりません。先に Load ステップを実行してください。")

    texts = df["text"].astype(str).tolist()

    # 2) Load HF model + tokenizer (CPU)
    model_name = "LiquidAI/LFM2-ColBERT-350M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    hf_model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    hf_model.eval()

    # 3) Mean pooling function (mask-aware)
    def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, T, 1)
        masked = last_hidden_state * mask
        summed = masked.sum(dim=1)  # (B, H)
        counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        return summed / counts

    # 4) Encode in batches to avoid OOM
    batch_size = 16
    pooled_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            outputs = hf_model(**inputs)
            token_embeddings = outputs.last_hidden_state  # (B, T, H)
            pooled = mean_pooling(token_embeddings, inputs["attention_mask"])  # (B, H)
            # L2 normalize
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            pooled_vecs.append(pooled.cpu().numpy())

    embeddings = np.vstack(pooled_vecs).astype(np.float32)  # (N, H)

    # 5) Build output DataFrame for DuckDB (embedding as list[float])
    out_df = pd.DataFrame(
        {
            "id": df["id"].astype(int),
            "text": df["text"].astype(str),
            "embedding": [row.astype(np.float32).tolist() for row in embeddings],
        }
    )
    return out_df
