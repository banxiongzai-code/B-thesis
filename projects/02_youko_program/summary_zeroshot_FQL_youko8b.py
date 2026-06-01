#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
temperature=0.3
top_p=0.95
repetition_penalty=1.2
"""

__author__ = "Ban Yuya"
__version__ = "3.0.0"
__date__ = "2025/06/30"

import json
import os
import gc
import sys
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_RETRY = 10
# PASSING_TOTAL_SCORE = 16  # 合計点の合格ライン
# REQUIRED_忠実性 = 4     # 正確性の最低ライン
# REQUIRED_LENGTH_SCORE = 4 # 文字数評価の最低ライン

def main():
    args = get_args()
    model, tokenizer = load_model(args.model)
    dataset = load_jl_dataset(args.input_file)
    os.makedirs(args.output_path, exist_ok=True)
    generate_with_summary(args, model, tokenizer, dataset)
    
def get_args():
    """
    コマンドライン引数を応答します
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        required=False,
        default="japanese_XLSum_v2.0/japanese_test.jsonl",
        help="入力ファイル名を指定します",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=False,
        default="output_zeroshot_FQL_youko8B",
        help="出力ディレクトリ名を指定します",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        default="rinna/llama-3-youko-8b-instruct",
        help="モデル名を指定します",
    )
    return parser.parse_args()

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def load_jl_dataset(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_json(filepath, data):
    """JSON保存用ヘルパー関数"""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# ------------------------------------------------------------------
#  LLM Generation Helper
# ------------------------------------------------------------------
def call_llm(model, tokenizer, prompt):
    """
    LLMを呼び出す汎用関数
    履歴を持たず、今回のプロンプトだけで生成を行う
    """
    messages = [
        {"role": "system", "content": "あなたは誠実で優秀な日本人のアシスタントです。常に日本語で回答してください。"},
        {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=1024,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.3, # 制御重視のため低めに設定
            top_p=0.95,
            repetition_penalty=1.2
        )
    
    response = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(response, skip_special_tokens=True).strip()

# # ------------------------------------------------------------------
# #  Step 1: Extraction (重要な部分を抜き出す)
# # ------------------------------------------------------------------
# def step_extract_keypoints(model, tokenizer, text):
#     """記事から重要ポイントを箇条書きで抽出する"""
#     prompt = (
#         "## 指示\n"
#         "以下の「入力文章」から、要約の核となる重要文を3つから5つ選定し、箇条書きで抽出してください。\n\n"
#         "# 抽出の基準\n"
#         "1. **事実優先**: 「誰が・いつ・何を・どうした」という具体的な事実が含まれる文を優先してください。\n"
#         "2. **網羅性の排除**: 細かい具体例や、挨拶、導入、繰り返し表現は選ばないでください。\n"
#         "3. **完全一致**: 文章を要約したり言い換えたりせず、元の文章から一言一句変えずにそのまま抜き出してください。\n\n"
#         f"## 入力文章\n{text}"
#     )
#     return call_llm(model, tokenizer, prompt)

# ------------------------------------------------------------------
#  Step 2: Drafting / Refining (Single Turn Logic)
# ------------------------------------------------------------------
def step_draft(model, tokenizer, original_text, target_length):
    # 初回生成用プロンプト
    prompt = (
        "## 指示\n"
        f"以下の「入力文章」に基づき、{target_length}文字ぴったりの日本語の要約文を作成してください。\n"
        "## 制約条件\n"
        "1. **情報の限定**: 「重要なポイント」に含まれない情報は一切追加せず、固有名詞や数字などの事実関係を正確に保つこと。\n"
        "2. **文体の自然さ**: 体言止め（名詞での文末終了）は禁止とし、「〜である」「〜した」等の適切な述語で文を結ぶこと。\n"
        "3. **構成の連続性**: 箇条書きや機械的な羅列は避け、接続詞を用いて文脈が滑らかに繋がる一つの文章（段落）にすること。\n"
        "4. **出力の浄化**: 要約文以外の挨拶、説明、タイトル等は一切出力しないこと。\n"
        f"## 入力文章\n{original_text}"
    )

    return call_llm(model, tokenizer, prompt)

# ------------------------------------------------------------------
#  Step 3: Judging
# ------------------------------------------------------------------
def step_judge(model, tokenizer, original_text, summary):
    # ユーザープロンプトに評価基準を埋め込む
    prompt = f"""
        ## 指示
        以下の「元文書」および「重要なポイント」を参照し、評価対象となる要約の品質を採点してください。

        ## 評価プロセス
        各評価項目について、以下の「定義」に基づき、1〜5点で採点を行ってください。
        個別の採点結果を統合し、要約全体の「良し悪し」と「修正すべき優先順位」を総括してください。
        **4点以上を合格（修正不要）とし、少しでも欠陥がある場合は3点以下をつけてください。**

        ## 評価基準
        1. **忠実性**
        - **定義**: 生成された要約の内容が、元文書に基づいているか。嘘（ハルシネーション）や、元記事にない情報の捏造がないか。
        - 5点: 完全に元文書の内容と一致している。
        - 3点: 元記事の内容は合っているが、数値や細かいニュアンスに誤りがある。
        - 1点: 重大な嘘や捏造が含まれている。

        2. **流暢性**
        - **定義**: 文と文のつながりが論理的で自然か。「体言止め」がなく、完結した文章になっているか。
        - 5点: 接続詞が適切に使われ、プロが書いたようにスムーズに読める。
        - 3点: 意味は通じるが、唐突な展開や、文末が不自然（途中で切れている等）な箇所がある。
        - 1点: 文法崩壊や、文章として成立していない。

        3. **網羅性**
        - **定義**: 入力として与えられた「重要なポイント」の内容が、要約に過不足なく反映されているか。
        - 5点: 全ての重要なポイントの要素が含まれている。
        - 3点: 重要なポイントのうち1つ程度の要素が欠落している。
        - 1点: 重要なポイントがほとんど反映されていない。

        4. **非冗長性**
        - **定義**: 文字数稼ぎのための「意味のない繰り返し」や「回りくどい表現」がないか。
        - 5点: 情報密度が高く、無駄な表現が一切ない。
        - 3点: 文字数を埋めるための冗長な表現や、同じ語句の繰り返しが見られる。
        - 1点: 同じ内容を何度も繰り返しており、読んでいて苦痛である。

        ## 出力形式 (JSON)
        必ず以下のJSON形式のみを出力してください。
        {{
            "品質評価": {{
                "忠実性": {{ "score": <1-5の整数>, "reason": "減点理由または評価根拠" }},
                "流暢性": {{ "score": <1-5の整数>, "reason": "減点理由または評価根拠" }},
                "網羅性": {{ "score": <1-5の整数>, "reason": "減点理由または評価根拠" }},
                "非冗長性": {{ "score": <1-5の整数>, "reason": "減点理由または評価根拠" }},
                "総合評価": {{ "reason": "評価根拠" }}
            }}
        }}

        ## 入力データ
        【元文書】
        {original_text}

        【評価対象の要約】
        {summary}
        """

    result_text = call_llm(model, tokenizer, prompt)
    
    # JSON抽出とパース
    try:
        # Markdownの```json ... ``` がある場合とない場合の両方に対応
        json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            # 2. クリーニング処理（パースエラーの原因を潰す）
            # Markdownのコードブロック記号を削除
            json_str = re.sub(r'```json\s*', '', json_str)
            json_str = re.sub(r'```', '', json_str)
            
            # 全角のダブルクォートを半角に置換
            json_str = json_str.replace('”', '"').replace('“', '"')
            
            # 全角のカンマを半角に置換
            json_str = json_str.replace('，', ',')
            
            # (念のため) 末尾の余計なカンマを削除する正規表現
            # 例: {"a": 1, } -> {"a": 1 }
            json_str = re.sub(r',\s*\}', '}', json_str)
            json_str = re.sub(r',\s*\]', ']', json_str)

            raw_data = json.loads(json_str)
            
            # 1. ネストされた "evaluations" を取得（なければ空辞書）
            evaluations = raw_data.get("品質評価", {})
            
            # 2. 結果を格納する辞書（フラットな形式）
            data = {}
            
            # 3. 取得したいキーのリスト (プロンプトとプログラムで一致している前提)
            target_keys = ["忠実性", "流暢性", "網羅性", "非冗長性"]
            
            for key in target_keys:
                # 該当項目のオブジェクトを取得 (例: {"score": 5, "reason": "..."})
                item = evaluations.get(key, {})
                
                # スコア抽出（取得できない、数値変換できない場合はデフォルト1）
                try:
                    data[key] = int(item.get("score", 1))
                except (ValueError, TypeError):
                    data[key] = 1
                
                # 個別の理由も保存しておく（デバッグや分析用）
                # 例: data['網羅性_reason']
                data[f"{key}_reason"] = item.get("reason", "理由なし")
            # 2. 点数がない項目（総合評価）はループの外で個別に取る
            overall_item = evaluations.get("総合評価", {})
            data["総合評価_reason"] = overall_item.get("reason", "総合評価なし")
            return data
            
        else:
            # パース失敗時はエラー情報を含む辞書を返す
            return {
                "忠実性": 1, "流暢性": 1, "網羅性": 1, "非冗長性": 1,
                "reason": "JSONが見つかりませんでした", 
                "error": "JSON not found", "raw": result_text
            }
            
    except Exception as e:
        # エラー時も辞書を返す（プログラムを止めないため）
        return {
            "忠実性": 1, "流暢性": 1, "網羅性": 1, "非冗長性": 1,
            "reason": f"パースエラー: {str(e)}", 
            "error": str(e), "raw": result_text
        }

# ------------------------------------------------------------------
#  Main Process Loop
# ------------------------------------------------------------------
def generate_with_summary(args, model, tokenizer, dataset):
    for i, article in enumerate(dataset):
        print(f"COUNT: {j}")    
        article_id = article.get("id", f"article_{i}")
        original_text = article["text"]
        target_length = len(article["summary"])
        print(f"Article {article_id}: number {j}")
        
        # 生成
        current_summary = step_draft(
            model, tokenizer, original_text, target_length
        )
        current_len = len(current_summary)
        
        # 評価
        print("  [2] Judging...")
        eval_result = step_judge(model, tokenizer, original_text, current_summary)
        # --- スコア計算と判定準備 ---
        quality_keys = ["忠実性", "流暢性", "網羅性", "非冗長性"]
        
        # パースエラーチェック
        if "error" in eval_result:
            print(f"      -> Judge Error: {eval_result['error']}")
            total_score = 0
        else:
            # 合計点計算 (20点満点)
            total_score = sum(eval_result.get(k, 0) for k in quality_keys)
        
        print(f"      -> Quality Score: {total_score}/20")

        # 保存
        save_json(os.path.join(args.output_path, f"{article_id}_zeroshot.json"), {
            "id": article_id,
            "target_length": target_length,
            "current_length": current_len,
            "summary": current_summary,
            "evaluation": eval_result,
            "total_score": total_score
        })
        
        # --- 合否判定ロジック (厳密なAND条件) ---
        # 2. 品質判定 (全項目が4点以上)
        # ※ get(k, 0) で取得失敗時は0点扱いでFalseにする
        is_quality_pass = all(eval_result.get(k, 0) >= 4 for k in quality_keys)
        
        if target_length == current_len and is_quality_pass:
            print("成功：目標文字数に一致しました。")
            break

if __name__ == "__main__":
    sys.exit(main())
