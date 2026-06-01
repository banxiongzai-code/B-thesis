"""
規定の要約文字数または10回目で終了したデータを自動で抜き取るプログラム
"""

__author__ = "Ban Yuya"
__version__ = "1.0.1"
__date__ = "2025/05/26 (Created: 2025/05/26)"
import os
import json
import shutil

def extract_target_summaries(input_directory, output_directory):
    """
    規定の要約文字数または10回目で終了したデータを抜き取るプログラム

    Args:
        input_directory (str): 入力ディレクトリのパス
        output_directory (str): 出力ディレクトリのパス
        target_length (int): 規定の要約文字数
    """
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 入力ディレクトリ内のファイルを取得
    all_files = [f for f in os.listdir(input_directory) if f.endswith(".json")]

    extracted_count = 0

    for file in all_files:
        file_path = os.path.join(input_directory, file)

        # JSONファイルを読み込み
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 必要なデータを判定 (規定の文字数か10回目か)
        if ("attempt" in data and "target_length" in data and "current_length" in data and "evaluation" in data):
            iteration = data["attempt"]
            summary_length = data["target_length"]
            output_length = data["current_length"]
            eval_scores = data["evaluation"]

            # 各評価項目のスコアが4点以上かどうかを判定するフラグ
            # (数値以外の値が含まれている可能性があるため、辞書から数値のものだけを取り出して判定)
            scores = [v for k, v in eval_scores.items() if isinstance(v, (int, float))]
            all_scores_above_4 = all(s >= 4 for s in scores) if scores else False

            # 条件:
            # A: iteration が 10 である
            # B: 文字数が一致している かつ 全スコアが4以上
            if iteration == 10 or (summary_length == output_length and all_scores_above_4):
                shutil.copy(file_path, os.path.join(output_directory, file))
                extracted_count += 1
                
                status = "10th iteration" if iteration == 10 else "Perfect Match & High Score"
                print(f"[{status}] File: {file}, target: {summary_length}, current: {output_length}")

    print(f"条件を満たすファイル {extracted_count} 件が {output_directory} にコピーされました。")

# 使用例
input_directory = "../output_summary/output_FQL_GPT5"
output_directory = "../output_summary/output_FQL_GPT5_perfect"
extract_target_summaries(input_directory, output_directory)
