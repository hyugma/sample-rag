ローカルRAGデモ（PyLate + HF Transformers + Gradio）

概要
- PyLate（ColBERT Late Interaction）で高精度なトークン単位検索を行い、取得した文脈を HF Transformers のチャットモデルに与えて回答を生成するローカル RAG デモです。
- ドキュメントは CSV（id, text）で管理し、検索用インデックスはローカルディレクトリ pylate-index/ に保存されます（.gitignore 済み）。
- 既定のモデル:
  - Retriever: LiquidAI/LFM2-ColBERT-350M
  - Generator: LiquidAI/LFM2-1.2B-RAG

クイックスタート
1) 依存関係を用意
- uv を利用する場合（推奨例）
  - macOS（Homebrew）: brew install uv
  - 依存同期: uv sync
- または任意の仮想環境で requirements 相当をインストールしてください（pyproject.toml 参照）。

2) アプリを起動
- uv run python app.py
  - ポートは 7860（GRADIO_SERVER_PORT/PORT で上書き可）
- もしくは python app.py

3) 使い方（Gradio UI）
- Data Prep タブ:
  - UIより任意のCSVをアップロード/編集し、保存（input.csvとして保存）
  - 「PyLate インデックスを再構築」をクリックしてinput.csvからインデックス生成
- RAG タブ:
  - 質問と TopK を指定して送信
  - 初回は生成モデルのダウンロードに同意（チェックボックス）すると自動取得します
  - 取得文脈・スコア・最終プロンプト（テンプレート適用済み）がログに表示されます

内部処理フロー（ハイレベル）
1) データ管理
- ユーザーが CSV（id, text）を用意 → input.csv
- 「インデックス再構築」で PLAID 形式のインデックスを pylate-index/ に生成
- id → 原文のマッピングを id2text.json に保存

2) 検索（Late Interaction）
- クエリを ColBERT エンコーダでベクトル化（is_query=True）
- PLAID インデックスから TopK を取得
- 取得 id を id2text.json で本文に引き当て

3) プロンプト組み立て
- 取得した文脈とユーザー質問から chat メッセージ配列を構築（system/user）
- tokenizer.apply_chat_template(messages, add_generation_prompt=True) でモデル固有のチャットテンプレートを適用
- ログには「Rendered prompt」としてテンプレート適用後の実体を出力

4) 生成
- apply_chat_template(..., tokenize=True, return_tensors="pt") でトークナイズした入力を model.generate に渡す
- 適切な EOS/PAD 設定で停止制御
- 生成テキストを decode して回答として返却

環境変数（主なもの）
- EMBED_MODEL_NAME（既定: LiquidAI/LFM2-ColBERT-350M）
- HF_CHAT_MODEL（既定: LiquidAI/LFM2-1.2B-RAG）
- TOP_K（既定: 1）
- GRADIO_SERVER_PORT または PORT（既定: 7860）
- PYLATE_INDEX_FOLDER（既定: pylate-index）
- PYLATE_INDEX_NAME（既定: index）

リポジトリ運用メモ
- pylate-index/、大きなモデルファイル（*.safetensors など）は .gitignore 済み
- input.csv は小規模サンプルであればコミット可（大きなデータは非推奨）
- 依存は pyproject.toml（uv での運用を想定）

クレジット
- PyLate: https://github.com/lightonai/pylate
- LFM2-ColBERT-350M: https://huggingface.co/LiquidAI/LFM2-ColBERT-350M
- LFM2-1.2B-RAG: https://huggingface.co/LiquidAI/LFM2-1.2B-RAG
