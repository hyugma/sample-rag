import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path
import json

import numpy as np
import pandas as pd
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

# PyLate (Late Interaction: ColBERT + PLAID index)
from pylate import indexes, models, retrieve

# ===== Settings =====
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "LiquidAI/LFM2-ColBERT-350M")
HF_CHAT_MODEL = os.environ.get("HF_CHAT_MODEL", "LiquidAI/LFM2-1.2B-RAG")
TOP_K = int(os.environ.get("TOP_K", "1"))
SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860")))

# PyLate Index settings
INDEX_FOLDER = os.environ.get("PYLATE_INDEX_FOLDER", "pylate-index")
INDEX_NAME = os.environ.get("PYLATE_INDEX_NAME", "index")
ENCODE_BATCH = int(os.environ.get("PYLATE_ENCODE_BATCH", "32"))

INPUT_CSV = Path("input.csv")

# ===== Device selection =====
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
DTYPE_GEN = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

# ===== PyLate（ColBERT）Retriever =====
colbert_model = models.ColBERT(model_name_or_path=EMBED_MODEL_NAME)
# Some tokenizers may miss pad_token; set it to eos_token if available
try:
    if getattr(colbert_model.tokenizer, "pad_token", None) is None and getattr(colbert_model.tokenizer, "eos_token", None) is not None:
        colbert_model.tokenizer.pad_token = colbert_model.tokenizer.eos_token
except Exception:
    pass

def _get_plaid_index(override: bool = False):
    """
    Return PyLate PLAID index. override=True recreates it.
    """
    return indexes.PLAID(
        index_folder=INDEX_FOLDER,
        index_name=INDEX_NAME,
        override=override,
    )

def _id2text_path() -> Path:
    return Path(INDEX_FOLDER) / "id2text.json"

def _load_id2text() -> dict[str, str]:
    p = _id2text_path()
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                m = json.load(f)
            # ensure str->str
            return {str(k): str(v) for k, v in m.items()}
        except Exception:
            return {}
    return {}

def _save_id2text(mapping: dict[str, str]) -> None:
    Path(INDEX_FOLDER).mkdir(parents=True, exist_ok=True)
    p = _id2text_path()
    with p.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False)

def rebuild_pylate_index() -> str:
    """
    Build/overwrite PyLate index from input.csv (expects id, text columns).
    Also writes id->text mapping to pylate-index/id2text.json for retrieval display.
    """
    if not INPUT_CSV.exists():
        return "エラー: input.csv が見つかりません。CSVを用意するか、Data Prepタブで作成してください。"
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        return f"input.csv の読み込みに失敗しました: {e}"

    if not {"id", "text"}.issubset(df.columns):
        return "エラー: input.csv は少なくとも 'id' と 'text' 列を含む必要があります。"

    df = df.dropna(subset=["id", "text"])
    try:
        ids = [str(int(i)) for i in df["id"].tolist()]
    except Exception:
        return "エラー: 'id' 列は整数に変換できる値である必要があります。"
    documents = df["text"].astype(str).tolist()

    if len(ids) == 0:
        return "エラー: 有効な行がありません（空のCSV）。"

    try:
        index = _get_plaid_index(override=True)  # overwrite existing index if any
        documents_embeddings = colbert_model.encode(
            documents,
            batch_size=ENCODE_BATCH,
            is_query=False,
            show_progress_bar=True,
        )
        index.add_documents(
            documents_ids=ids,
            documents_embeddings=documents_embeddings,
        )
        # save id->text mapping
        _save_id2text({i: t for i, t in zip(ids, documents)})
        return f"PyLateインデックス再構築完了: {len(documents)} 文書 -> {INDEX_FOLDER}/{INDEX_NAME}"
    except Exception as e:
        return f"インデックス構築中にエラー: {e}"

def _fetch_texts_by_ids(ids: list[str]) -> list[str]:
    """
    Map retrieved ids to texts using sidecar mapping under pylate-index/id2text.json.
    """
    if not ids:
        return []
    mapping = _load_id2text()
    return [mapping.get(str(i), "") for i in ids]

# ===== Generator (HF) =====
chat_tokenizer = None
chat_model = None

def ensure_chat_model(consent_download: bool) -> str | None:
    """
    Prepare HF generator model.
    - Try local only (local_files_only=True)
    - If not found and consent_download=True, fetch from network
    - Return error string on failure, or None on success
    """
    global chat_tokenizer, chat_model
    if chat_model is not None and chat_tokenizer is not None:
        return None

    try:
        chat_tokenizer = AutoTokenizer.from_pretrained(HF_CHAT_MODEL, trust_remote_code=True, local_files_only=True)
        chat_model = AutoModelForCausalLM.from_pretrained(
            HF_CHAT_MODEL,
            trust_remote_code=True,
            local_files_only=True,
            dtype=DTYPE_GEN,
        )
        chat_model.to(DEVICE).eval()
        try:
            if getattr(chat_tokenizer, "pad_token_id", None) is None and getattr(chat_tokenizer, "eos_token_id", None) is not None:
                chat_tokenizer.pad_token = chat_tokenizer.eos_token
        except Exception:
            pass
        try:
            if getattr(chat_model, "generation_config", None) is not None and getattr(chat_model.generation_config, "pad_token_id", None) is None and getattr(chat_tokenizer, "eos_token_id", None) is not None:
                chat_model.generation_config.pad_token_id = chat_tokenizer.eos_token_id
        except Exception:
            pass
        return None
    except Exception as e_local:
        if not consent_download:
            return (
                "生成用のHFモデルがローカルに見つかりません。画面の「未ダウンロードならHugging Faceからモデルをダウンロードしてよい」にチェックを入れて再実行してください。\n"
                f"モデル名: {HF_CHAT_MODEL}\n詳細: {e_local}"
            )
        try:
            chat_tokenizer = AutoTokenizer.from_pretrained(HF_CHAT_MODEL, trust_remote_code=True)
            chat_model = AutoModelForCausalLM.from_pretrained(
                HF_CHAT_MODEL,
                trust_remote_code=True,
                dtype=DTYPE_GEN,
            )
            chat_model.to(DEVICE).eval()
            try:
                if getattr(chat_tokenizer, "pad_token_id", None) is None and getattr(chat_tokenizer, "eos_token_id", None) is not None:
                    chat_tokenizer.pad_token = chat_tokenizer.eos_token
            except Exception:
                pass
            try:
                if getattr(chat_model, "generation_config", None) is not None and getattr(chat_model.generation_config, "pad_token_id", None) is None and getattr(chat_tokenizer, "eos_token_id", None) is not None:
                    chat_model.generation_config.pad_token_id = chat_tokenizer.eos_token_id
            except Exception:
                pass
            return None
        except Exception as e_dl:
            return f"Hugging Faceからのモデルダウンロード/ロードに失敗しました: {e_dl}"

def build_prompt(messages: list[dict]) -> str:
    """
    Fallback simple prompt build if no chat template.
    """
    system = ""
    user = ""
    if messages:
        if messages[0].get("role") == "system":
            system = messages[0].get("content", "")
        user = messages[-1].get("content", "")
    prompt = f"{system}\n\nUser: {user}\nAssistant:"
    return prompt


def render_chat_prompt(messages: list[dict]) -> str:
    """
    Render messages using the model's chat template if available, otherwise fallback.
    """
    try:
        return chat_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return build_prompt(messages)

def generate_with_hf_chat(messages: list[dict]) -> str:
    """
    Generate using tokenizer.apply_chat_template() if available.
    Falls back to plain prompt tokenization on failure.
    """
    try:
        model_inputs = chat_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(DEVICE)
    except Exception:
        prompt = build_prompt(messages)
        model_inputs = chat_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Normalize inputs to a mapping that can be expanded with ** for generate()
    if isinstance(model_inputs, torch.Tensor):
        inputs = {"input_ids": model_inputs}
    else:
        # BatchEncoding or dict-like
        inputs = model_inputs

    # Ensure attention_mask is present to avoid pad/eos ambiguity warnings
    if "attention_mask" not in inputs:
        try:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long, device=inputs["input_ids"].device)
        except Exception:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

    # Collect reasonable EOS token ids (model-specific special tokens included if present)
    eos_ids = set()
    if getattr(chat_tokenizer, "eos_token_id", None) is not None:
        eos_ids.add(chat_tokenizer.eos_token_id)
    for tok in ("<|eot_id|>", "<|im_end|>", "<|end|>"):
        try:
            tid = chat_tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.add(tid)
        except Exception:
            pass

    gen_kwargs = dict(
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=chat_tokenizer.eos_token_id if getattr(chat_tokenizer, "eos_token_id", None) is not None else None,
    )
    if eos_ids:
        gen_kwargs["eos_token_id"] = list(eos_ids)

    with torch.no_grad():
        output_ids = chat_model.generate(**inputs, **gen_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[0][input_len:]
    text = chat_tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()

def stream_generate_with_hf_chat(messages: list[dict]):
    """
    Stream tokens using TextIteratorStreamer. Yields incremental decoded text.
    """
    try:
        model_inputs = chat_tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(DEVICE)
    except Exception:
        prompt = build_prompt(messages)
        model_inputs = chat_tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Normalize inputs into mapping for generate()
    if isinstance(model_inputs, torch.Tensor):
        inputs = {"input_ids": model_inputs}
    else:
        inputs = model_inputs

    # Ensure attention_mask is present to avoid pad/eos ambiguity warnings
    if "attention_mask" not in inputs:
        try:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long, device=inputs["input_ids"].device)
        except Exception:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)

    # EOS/PAD setup
    eos_ids = set()
    if getattr(chat_tokenizer, "eos_token_id", None) is not None:
        eos_ids.add(chat_tokenizer.eos_token_id)
    for tok in ("<|eot_id|>", "<|im_end|>", "<|end|>"):
        try:
            tid = chat_tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0:
                eos_ids.add(tid)
        except Exception:
            pass

    gen_kwargs = dict(
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=chat_tokenizer.eos_token_id if getattr(chat_tokenizer, "eos_token_id", None) is not None else None,
    )
    if eos_ids:
        gen_kwargs["eos_token_id"] = list(eos_ids)

    # Create streamer and launch generation in background
    streamer = TextIteratorStreamer(chat_tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {**inputs, **gen_kwargs, "streamer": streamer}
    thread = Thread(target=chat_model.generate, kwargs=gen_kwargs)
    thread.start()

    accumulated = ""
    for piece in streamer:
        accumulated += piece
        yield accumulated

def _build_messages(context: str, user_query: str):
    sys = (
        "You are a helpful assistant. Use the provided context to answer the user's question. "
        "If the answer cannot be found in the context, say you do not know."
    )
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {user_query}\n"
        f"Answer in Japanese if the user asked in Japanese."
    )
    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": user},
    ]

# ===== Retrieval（PyLate index） =====
def _retrieve_with_pylate(query: str, top_k: int = TOP_K):
    """
    Retrieve top-k document texts and scores using PyLate (ColBERT + PLAID index).
    """
    try:
        index = _get_plaid_index(override=False)
        retriever = retrieve.ColBERT(index=index)
    except Exception as e:
        raise RuntimeError(f"インデックスの読み込みに失敗しました。先に『PyLate インデックスを再構築』を実行してください。詳細: {e}")

    try:
        queries_embeddings = colbert_model.encode(
            [query],
            batch_size=ENCODE_BATCH,
            is_query=True,
            show_progress_bar=False,
        )
        scores_all = retriever.retrieve(
            queries_embeddings=queries_embeddings,
            k=int(top_k),
        )
        results = scores_all[0] if scores_all else []
        ids = [str(r["id"]) for r in results]
        scores = np.asarray([float(r["score"]) for r in results], dtype=np.float32)
        texts = _fetch_texts_by_ids(ids)
        return texts, scores
    except Exception as e:
        raise RuntimeError(f"検索中にエラーが発生しました: {e}")

# ===== RAG infer =====
def rag_infer(user_query: str, top_k: int, consent_download: bool):
    user_query = (user_query or "").strip()
    if not user_query:
        return "質問を入力してください。", ""

    try:
        retrieved_texts, top_scores = _retrieve_with_pylate(user_query, top_k=int(top_k))
    except Exception as e:
        return (
            "検索中にエラーが発生しました。input.csv を用意し、"
            "『PyLate インデックスを再構築』を実行してください。\n"
            f"詳細: {e}",
            ""
        )

    context = "\n\n".join(retrieved_texts)
    messages = _build_messages(context, user_query)

    # Ensure model/tokenizer are available
    err = ensure_chat_model(consent_download=consent_download)

    # Prepare logs (render prompt via chat template if tokenizer is available)
    try:
        scores_list = np.round(np.asarray(top_scores), 4).tolist()
    except Exception:
        scores_list = []
    if chat_tokenizer is not None:
        try:
            rendered_prompt = render_chat_prompt(messages)
        except Exception:
            rendered_prompt = build_prompt(messages)
    else:
        rendered_prompt = build_prompt(messages)

    log_lines = [
        f"Device: {DEVICE}, dtype: {DTYPE_GEN}",
        f"Retriever (PyLate): {EMBED_MODEL_NAME} with PLAID index",
        f"Generator: {HF_CHAT_MODEL}",
        f"TopK: {int(top_k)}",
        f"Query: {user_query}",
        f"Scores: {scores_list}",
        "Retrieved texts:",
    ]
    for i, txt in enumerate(retrieved_texts, 1):
        log_lines.append(f"{i}) {txt}")
    log_lines.append("Rendered prompt (via chat template if available):")
    log_lines.append(rendered_prompt)
    logs = "\n".join(log_lines)

    if err:
        # Stream-compatible early exit
        yield err, logs
        return

    # Stream tokens using tokenizer.apply_chat_template()-based inputs
    try:
        for partial in stream_generate_with_hf_chat(messages):
            yield partial, logs
    except Exception as e_gen:
        yield f"生成中にエラーが発生しました: {e_gen}", logs

# ===== Data Prep（CSVアップロード/編集/保存 → PyLateインデックス再構築） =====
def _ensure_dataframe(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    try:
        return pd.DataFrame(df)
    except Exception:
        return pd.DataFrame(columns=["id", "text"])

def load_uploaded_csv(file_path: str) -> pd.DataFrame:
    """アップロードCSVを読み込んで表示用DataFrameを返す"""
    try:
        if not file_path:
            return pd.DataFrame(columns=["id", "text"])
        df = pd.read_csv(file_path)
        return df
    except Exception:
        return pd.DataFrame(columns=["id", "text"])

def load_current_csv() -> pd.DataFrame:
    """プロジェクト内の input.csv を読み込んで表示。存在しなければ空DF"""
    if INPUT_CSV.exists():
        try:
            df = pd.read_csv(INPUT_CSV)
            return df
        except Exception:
            return pd.DataFrame(columns=["id", "text"])
    return pd.DataFrame(columns=["id", "text"])

def save_csv_and_load_df(df) -> str:
    """編集結果のDataFrameを input.csv に保存"""
    df = _ensure_dataframe(df)
    if not {"id", "text"}.issubset(df.columns):
        return "エラー: DataFrameは少なくとも 'id' と 'text' 列を含む必要があります。"

    df = df.dropna(subset=["id", "text"])
    try:
        df["id"] = df["id"].astype(int)
    except Exception:
        return "エラー: 'id' 列は整数に変換できる必要があります。"

    try:
        df.to_csv(INPUT_CSV, index=False, encoding="utf-8")
    except Exception as e:
        return f"input.csv の保存に失敗しました: {e}"

    return f"保存完了: input.csv（{len(df)}行）。次に『PyLate インデックスを再構築』を実行してください。"

# ===== Gradio app =====
with gr.Blocks(title="ローカルRAGデモ（PyLate + HF Transformers）") as demo:
    with gr.Tabs():
        with gr.Tab("Data Prep"):
            gr.Markdown(
                """
                ## CSV前処理（アップロード/閲覧/編集 → input.csv 保存 → PyLateインデックス）
                - CSVをアップロードして内容を確認・編集できます（最低限 id, text 列が必要）
                - 「保存」で input.csv を更新
                - 「PyLate インデックスを再構築」で検索用インデックスを再生成（pylate-index/）
                """
            )
            with gr.Row():
                up = gr.File(label="CSVアップロード（.csv）", file_types=[".csv"], type="filepath")
                #load_current_btn = gr.Button("現在の input.csv を読み込み")

            df_comp = gr.Dataframe(
                headers=["id", "text"],
                interactive=True,
                row_count=(3, "dynamic"),
                label="CSV内容（編集可）"
            )

            with gr.Row():
                save_btn = gr.Button("保存")
                index_btn = gr.Button("PyLate インデックスを再構築")

            prep_status = gr.Textbox(label="前処理ステータス/メッセージ", lines=6)
            index_logs = gr.Textbox(label="インデックス構築ログ", lines=12)

            # Events
            up.change(load_uploaded_csv, inputs=up, outputs=df_comp)
            #load_current_btn.click(load_current_csv, inputs=None, outputs=df_comp)
            save_btn.click(save_csv_and_load_df, inputs=df_comp, outputs=prep_status)
            index_btn.click(rebuild_pylate_index, inputs=None, outputs=index_logs)

        with gr.Tab("RAG"):
            gr.Markdown(
                f"""
                # ローカルRAGデモ（PyLate + HF Transformers）
                - Retriever (Late Interaction): {EMBED_MODEL_NAME} with PyLate PLAID index
                - Generator: {HF_CHAT_MODEL}
                - 生成用モデルはローカルに無い場合、下のチェックボックスで同意した上でHugging Faceからダウンロードして利用します。
                """
            )
            with gr.Row():
                with gr.Column(scale=2):
                    inp = gr.Textbox(label="質問を入力してください", lines=4)
                    consent = gr.Checkbox(
                        label="未ダウンロードならHugging Faceからモデルをダウンロードしてよい（初回のみ大容量DLの可能性あり）",
                        value=False,
                    )
                    with gr.Row():
                        topk = gr.Number(label="TopK (取得件数)", value=TOP_K, precision=0)
                        btn = gr.Button("送信")

                with gr.Column(scale=2):
                    out = gr.Textbox(label="回答", lines=12)
            with gr.Row():
                logs = gr.Textbox(label="内部ログ（取得コンテキスト/スコア/プロンプト）", lines=14)
            btn.click(rag_infer, inputs=[inp, topk, consent], outputs=[out, logs])


if __name__ == "__main__":
    # Allow overriding port via GRADIO_SERVER_PORT/PORT env
    demo.launch(server_port=SERVER_PORT)
