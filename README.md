ローカルRAGデモ: ELT(Extract, Load, Transform) + dbt + DuckDB + HF Transformers + Gradio

概要
- 目的: ELT処理（E: CSV, L: DuckDB, T: dbtによる埋め込み生成）を実演し、その結果をRAGアプリで検索・生成に利用する最小構成のデモ。
- 動作環境: Windows / macOS / Linux（Python 3.10+）
- 主要技術:
  - データベース: DuckDB
  - トランスフォーム: dbt（Pythonモデル: dbt-py）
  - Embeddingモデル: LiquidAI/LFM2-ColBERT-350M（Hugging Face）
  - 生成モデル: HF Transformers（既定: LiquidAI/LFM2-1.2B-RAG）※GGUF/llama-cppは不要
  - Webアプリ: Gradio
  - 検索: Transformers + NumPy（コサイン類似度）

セットアップ（uv + uv sync 推奨）

0. uv のインストール
- macOS（Homebrew）:
  - brew install uv
- macOS/Linux（公式インストーラ）:
  - curl -LsSf https://astral.sh/uv/install.sh | sh
- Windows（PowerShell 管理者）:
  - irm https://astral.sh/uv/install.ps1 | iex
- 補足: インストール後、必要に応じて PATH を更新

1. Python の用意（例: 3.12）
- uv python install 3.12

2. 依存関係の同期（.venv を自動作成）
- uv sync
  - 本リポジトリの pyproject.toml と uv.lock を元に .venv を作成し、依存を同期します
  - 以降の実行は uv run ... でOK（手動でvenvを有効化する必要はありません）

プロジェクト構成（主要）
- pyproject.toml（uv/uv sync 用の依存定義、スクリプト定義）
- input.csv（サンプル/編集対象のCSV）
- load_to_duckdb.py（CSV → DuckDB: raw_documents）
- rag_elt.db（スクリプト/変換後に生成）
- rag_elt_dbt/
  - dbt_project.yml
  - models/
    - transform_with_embeddings.py（Transformers + 平均プーリングで埋め込み生成）
    - raw_documents.sql（任意のラップ用）
- profiles.yml（dbt-duckdb 設定: path=rag_elt.db）
- app.py（Gradioアプリ：RAG/前処理/SQL Explorer）

Step 1: CSV（E: Extract）
- input.csv を用意（最低限 id, text 列）
- サンプル:
  ```
  id,text
  1,"DuckDBはシングルファイルで動作する高速な分析データベースです。"
  2,"dbtはSQLやPythonを使って変換処理をモデル化できるデータ変換フレームワークです。"
  3,"RAGは検索で得た文脈をLLMに与えて回答精度を高める手法です。"
  ```

Step 2: DuckDBへロード（L: Load）
- uv run python load_to_duckdb.py
  - load_to_duckdb.py が input.csv を読み込み、rag_elt.db に raw_documents を再作成します

Step 3: dbt で埋め込み生成（T: Transform）
- プロファイル（profiles.yml）例:
  ```
  rag_elt_dbt:
    target: dev
    outputs:
      dev:
        type: duckdb
        path: rag_elt.db
        threads: 4
        schema: main
        settings:
          memory_limit: "1GB"
  ```
- 変換実行:
  - uv run dbt debug --project-dir rag_elt_dbt --profiles-dir .
  - uv run dbt run --project-dir rag_elt_dbt --profiles-dir .
- 結果:
  - rag_elt.db に final_documents（id, text, embedding=list[float]）が作成されます
- 実装メモ:
  - models/transform_with_embeddings.py は Transformers 経由で LiquidAI/LFM2-ColBERT-350M を使用し、平均プーリング＋L2正規化でベクトル化します（sentence-transformers の 'MaxSim' 問題を回避）

Step 4: RAG + Gradio アプリ
- 起動:
  - uv run python app.py               # 既定ポート
  - PORT=7868 uv run python app.py     # ポートを指定したい場合の例
- UI 機能:
  - タブ「RAG」:
    - TopK（取得件数）: 数値入力（上下ステッパー）
    - 内部ログ: 取得コンテキスト、スコア、LLMへの最終プロンプトを表示
    - 生成モデルはローカル未所持時、チェックボックスで同意するとHFから自動ダウンロード
  - タブ「Data Prep」:
    - CSVアップロード/編集 → input.csv 保存 → DuckDBへ raw_documents 再ロード
    - 「dbt 変換を実行」で final_documents を再生成
  - タブ「SQL Explorer」:
    - テーブル一覧の更新、テンプレート挿入（SELECT * FROM <table> LIMIT 50）
    - 任意SQLの実行結果表示

コマンド例（スクリプトエイリアスなし）
- uv run python load_to_duckdb.py
- uv run dbt debug --project-dir rag_elt_dbt --profiles-dir .
- uv run dbt run --project-dir rag_elt_dbt --profiles-dir .
- uv run python app.py
- PORT=7860 uv run python app.py
- PORT=7868 uv run python app.py

トラブルシューティング
- uv が見つからない:
  - macOS: brew install uv
  - curl/PowerShell のインストーラで再インストール
- Transformers のモデルダウンロードが遅い/失敗する:
  - ネットワーク環境を確認
  - アクセス制限のあるモデルは Hugging Face のトークン設定が必要な場合あり
- dbt が rag_elt.db に接続できない:
  - profiles.yml の path が rag_elt.db を指していること
  - uv run dbt debug --project-dir rag_elt_dbt --profiles-dir . で検証
- メモリ/速度:
  - 生成器は環境に応じて自動で dtype を選択（CUDA/MPSなら float16、その他は float32）
  - Apple Silicon の場合は MPS を自動利用

変更履歴（セットアップの uv 化）
- 以前の venv + pip 手順は廃止し、uv + uv sync を標準手順に変更
- pyproject.toml で依存を管理し、uv run -s スクリプトで操作を簡略化
- GGUF/llama-cpp の事前配置は不要（生成は HF Transformers 経由）。未所持時は同意ダウンロードで対応
