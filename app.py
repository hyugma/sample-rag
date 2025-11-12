import os
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# グローバル設定（環境変数で上書き可）
DUCKDB_PATH = os.environ.get("RAG_ELT_DB_PATH", "rag_elt.db")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "LiquidAI/LFM2-ColBERT-350M")
HF_CHAT_MODEL = os.environ.get("HF_CHAT_MODEL", "LiquidAI/LFM2-1.2B-RAG")
TOP_K = int(os.environ.get("TOP_K", "1"))
SERVER_PORT = int(os.environ.get("PORT", "7860"))

# デバイス選択（Macの場合はMPSを優先）
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
DTYPE_GEN = torch.float16 if DEVICE in ("cuda", "mps") else torch.float32

# ===== Embedding（Retriever）モデルのロード（起動時に一度） =====
# ColBERTエンコーダをTransformersで直接ロードし、平均プーリングでベクトル化
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
hf_model = AutoModel.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
hf_model.eval()

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

def encode_query(query: str) -> np.ndarray:
    inputs = tokenizer(
        [query],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = hf_model(**inputs)
        token_embeddings = outputs.last_hidden_state
        pooled = mean_pooling(token_embeddings, inputs["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled[0].cpu().numpy().astype(np.float32)

# ===== 生成（Generator）用HFモデル（同意後にダウンロード可） =====
chat_tokenizer = None
chat_model = None

def ensure_chat_model(consent_download: bool) -> str | None:
    """
    生成用HFモデルを準備する。
    - まずローカルのみでロード（local_files_only=True）
    - 見つからず consent_download=True の場合はネットからダウンロードしてロード
    - エラー時はメッセージ文字列を返す。成功時はNone
    """
    global chat_tokenizer, chat_model
    if chat_model is not None and chat_tokenizer is not None:
        return None

    # まずローカルファイルのみでロードを試す
    try:
        chat_tokenizer = AutoTokenizer.from_pretrained(HF_CHAT_MODEL, trust_remote_code=True, local_files_only=True)
        chat_model = AutoModelForCausalLM.from_pretrained(
            HF_CHAT_MODEL,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=DTYPE_GEN,
        )
        chat_model.to(DEVICE).eval()
        return None
    except Exception as e_local:
        if not consent_download:
            return (
                "生成用のHFモデルがローカルに見つかりません。画面の「未ダウンロードならHugging Faceからモデルをダウンロードしてよい」にチェックを入れて再実行してください。\n"
                f"モデル名: {HF_CHAT_MODEL}\n詳細: {e_local}"
            )
        # 同意あり → ネットからダウンロードしてロード
        try:
            chat_tokenizer = AutoTokenizer.from_pretrained(HF_CHAT_MODEL, trust_remote_code=True)
            chat_model = AutoModelForCausalLM.from_pretrained(
                HF_CHAT_MODEL,
                trust_remote_code=True,
                torch_dtype=DTYPE_GEN,
            )
            chat_model.to(DEVICE).eval()
            return None
        except Exception as e_dl:
            return f"Hugging Faceからのモデルダウンロード/ロードに失敗しました: {e_dl}"

def build_prompt(messages: list[dict]) -> str:
    """
    Chatテンプレートが無い場合に備え、system+userから素朴なプロンプトを構築する。
    """
    system = ""
    user = ""
    if messages:
        if messages[0].get("role") == "system":
            system = messages[0].get("content", "")
        user = messages[-1].get("content", "")
    prompt = f"{system}\n\nUser: {user}\nAssistant:"
    return prompt

def generate_with_hf(prompt: str) -> str:
    inputs = chat_tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output_ids = chat_model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            do_sample=False,
            pad_token_id=chat_tokenizer.eos_token_id,
        )
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = output_ids[0][input_len:]
    text = chat_tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()

# ===== Retrieval（DuckDB内のfinal_documentsから検索） =====
def _load_final_documents():
    """DuckDBからfinal_documentsを読み込み、(ids, texts, normalized_embeddings) を返す"""
    con = duckdb.connect(DUCKDB_PATH, read_only=True)
    try:
        df = con.execute("SELECT id, text, embedding FROM final_documents").df()
    finally:
        con.close()

    if df.empty:
        raise RuntimeError("final_documents が空です。ELTのT（dbt run）を先に実行してください。")

    # embedding(list[float]) -> np.ndarray (N, D)
    embs = [np.asarray(e, dtype=np.float32) for e in df["embedding"].tolist()]
    E = np.vstack(embs)
    # 正規化（コサイン類似度計算を内積で行うため）
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)

    return df["id"].values, df["text"].astype(str).values, E

def _retrieve(query: str, top_k: int = TOP_K):
    """クエリを埋め込み化し、final_documents から TopK テキストとスコアを返す"""
    ids, texts, E = _load_final_documents()
    q = encode_query(query)
    scores = E @ q  # コサイン類似度（正規化済みの内積）
    top_idx = np.argsort(-scores)[:top_k]
    retrieved = [texts[i] for i in top_idx]
    top_scores = scores[top_idx]
    return retrieved, top_scores

def _build_messages(context: str, user_query: str):
    """簡易チャットプロンプトの構築"""
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

# ===== RAG infer =====
def rag_infer(user_query: str, top_k: int, consent_download: bool):
    user_query = (user_query or "").strip()
    if not user_query:
        return "質問を入力してください。", ""

    try:
        retrieved_texts, top_scores = _retrieve(user_query, top_k=int(top_k))
    except Exception as e:
        return (
            "検索中にエラーが発生しました。ELTのT（dbt run）を実施したか、"
            "DuckDB内に final_documents が存在するかをご確認ください。\n"
            f"詳細: {e}",
            ""
        )

    context = "\n\n".join(retrieved_texts)
    messages = _build_messages(context, user_query)

    # 生成モデルを準備（ローカル優先、未ダウンロード時は同意があれば取得）
    err = ensure_chat_model(consent_download=consent_download)
    prompt = build_prompt(messages)

    # ログの構築（モデル準備エラーでも提示する）
    try:
        scores_list = np.round(np.asarray(top_scores), 4).tolist()
    except Exception:
        scores_list = []
    log_lines = [
        f"Device: {DEVICE}, dtype: {DTYPE_GEN}",
        f"Retriever: {EMBED_MODEL_NAME}",
        f"Generator: {HF_CHAT_MODEL}",
        f"TopK: {int(top_k)}",
        f"Query: {user_query}",
        f"Scores: {scores_list}",
        "Retrieved texts:",
    ]
    for i, txt in enumerate(retrieved_texts, 1):
        log_lines.append(f"{i}) {txt}")
    log_lines.append("Prompt:")
    log_lines.append(prompt)
    logs = "\n".join(log_lines)

    if err:
        return err, logs

    try:
        answer = generate_with_hf(prompt)
    except Exception as e_gen:
        return f"生成中にエラーが発生しました: {e_gen}", logs

    return answer, logs

# ===== Data Prep（CSVアップロード/編集/保存→DuckDBロード、dbt変換） =====
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
    except Exception as e:
        # エラー時は空DFを返し、ユーザーはログで確認
        return pd.DataFrame(columns=["id", "text"])

def load_current_csv() -> pd.DataFrame:
    """プロジェクト内の input.csv を読み込んで表示。存在しなければ空DF"""
    p = Path("input.csv")
    if p.exists():
        try:
            df = pd.read_csv(p)
            return df
        except Exception:
            return pd.DataFrame(columns=["id", "text"])
    return pd.DataFrame(columns=["id", "text"])

def save_csv_and_load_df(df) -> str:
    """編集結果のDataFrameを input.csv に保存し、DuckDBに raw_documents としてロード"""
    df = _ensure_dataframe(df)
    # 必須列チェック
    if not {"id", "text"}.issubset(df.columns):
        return "エラー: DataFrameは少なくとも 'id' と 'text' 列を含む必要があります。"

    # 空/NaN行を除去し型整形
    df = df.dropna(subset=["id", "text"])
    try:
        df["id"] = df["id"].astype(int)
    except Exception:
        return "エラー: 'id' 列は整数に変換できる必要があります。"

    # 保存
    try:
        df.to_csv("input.csv", index=False)
    except Exception as e:
        return f"input.csv の保存に失敗しました: {e}"

    # DuckDBへロード
    try:
        con = duckdb.connect(DUCKDB_PATH)
        con.execute("DROP TABLE IF EXISTS raw_documents")
        con.register("df_src", df)
        con.execute(
            """
            CREATE TABLE raw_documents AS
            SELECT CAST(id AS BIGINT) AS id, CAST(text AS VARCHAR) AS text
            FROM df_src
            """
        )
        con.unregister("df_src")
        con.close()
    except Exception as e:
        return f"DuckDBへのロードに失敗しました: {e}"

    return f"保存・ロード完了: input.csv（{len(df)}行）→ {DUCKDB_PATH}::raw_documents"

def run_dbt_transform() -> str:
    """dbt run を実行し、ログを返す"""
    try:
        # 直接 run のみ（必要なら debug を追加可能）
        proc = subprocess.run(
            ["dbt", "run", "--project-dir", "rag_elt_dbt", "--profiles-dir", "."],
            capture_output=True,
            text=True,
        )
        logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
        if proc.returncode == 0:
            return "dbt run 成功\n" + logs
        else:
            return "dbt run 失敗\n" + logs
    except Exception as e:
        return f"dbt 実行時エラー: {e}"

# ===== DuckDB SQL Explorer helpers =====
def list_duckdb_tables():
    try:
        con = duckdb.connect(DUCKDB_PATH, read_only=True)
        df = con.execute(
            "SELECT table_schema, table_name, table_type "
            "FROM information_schema.tables "
            "WHERE table_schema NOT IN ('information_schema') "
            "ORDER BY table_schema, table_name"
        ).df()
        con.close()
        names = []
        for _, r in df.iterrows():
            schema = r["table_schema"]
            name = r["table_name"]
            full = f"{schema}.{name}" if schema and schema != "main" else name
            names.append(full)
        return names
    except Exception:
        return []

def run_duckdb_sql(sql: str):
    sql = (sql or "").strip()
    if not sql:
        return pd.DataFrame(), "SQLを入力してください。"
    try:
        con = duckdb.connect(DUCKDB_PATH, read_only=True)
        df = con.execute(sql).df()
        con.close()
        return df, f"OK: {len(df)} rows x {(len(df.columns) if not df.empty else 0)} cols"
    except Exception as e:
        return pd.DataFrame(), f"エラー: {e}"

def refresh_table_choices():
    try:
        return gr.update(choices=list_duckdb_tables())
    except Exception:
        return gr.update(choices=[])

# ===== Gradioアプリ =====
with gr.Blocks(title="ローカルRAGデモ（ELT + dbt + DuckDB + HF Transformers）") as demo:
    with gr.Tabs():
        with gr.Tab("RAG"):
            gr.Markdown(
                f"""
                # ローカルRAGデモ（ELT + dbt + DuckDB + HF Transformers）
                - Retriever: {EMBED_MODEL_NAME}
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

        with gr.Tab("Data Prep"):
            gr.Markdown(
                """
                ## CSV前処理（アップロード/閲覧/編集 → DuckDBロード → 変換実行）
                - CSVをアップロードして内容を確認・編集できます（最低限 id, text 列が必要）
                - 「保存してDuckDBへロード」で raw_documents を再作成
                - 「dbt 変換を実行」で final_documents を再生成
                """
            )
            with gr.Row():
                up = gr.File(label="CSVアップロード（.csv）", file_types=[".csv"], type="filepath")
                load_current_btn = gr.Button("現在の input.csv を読み込み")

            df_comp = gr.Dataframe(
                headers=["id", "text"],
                interactive=True,
                row_count=(3, "dynamic"),
                label="CSV内容（編集可）"
            )

            with gr.Row():
                save_btn = gr.Button("保存してDuckDBへロード")
                transform_btn = gr.Button("dbt 変換を実行")

            prep_status = gr.Textbox(label="前処理ステータス/メッセージ", lines=6)
            transform_logs = gr.Textbox(label="dbt実行ログ", lines=12)

            # イベント
            up.change(load_uploaded_csv, inputs=up, outputs=df_comp)
            load_current_btn.click(load_current_csv, inputs=None, outputs=df_comp)
            save_btn.click(save_csv_and_load_df, inputs=df_comp, outputs=prep_status)
            transform_btn.click(run_dbt_transform, inputs=None, outputs=transform_logs)

        with gr.Tab("SQL Explorer"):
            gr.Markdown(
                """
                ## DuckDB SQL Explorer
                - テーブルを選択してテンプレートを挿入、または自由にSQLを入力して実行
                - 接続先: RAG_ELT_DB_PATH（既定: rag_elt.db）
                """
            )
            with gr.Row():
                refresh_btn = gr.Button("テーブル一覧を更新")
                table_dd = gr.Dropdown(choices=list_duckdb_tables(), label="テーブル", interactive=True)
                insert_btn = gr.Button("テンプレート挿入")
            sql_box = gr.Textbox(label="SQLクエリ", lines=6, value="SELECT * FROM final_documents LIMIT 10")
            run_btn = gr.Button("SQLを実行")
            sql_df = gr.Dataframe(interactive=False, label="結果")
            sql_status = gr.Textbox(label="ステータス/メッセージ", lines=4)

            refresh_btn.click(refresh_table_choices, inputs=None, outputs=table_dd)
            insert_btn.click(lambda t: (f"SELECT * FROM {t} LIMIT 50" if t else ""), inputs=table_dd, outputs=sql_box)
            run_btn.click(run_duckdb_sql, inputs=sql_box, outputs=[sql_df, sql_status])


if __name__ == "__main__":
    demo.launch(server_port=SERVER_PORT)
