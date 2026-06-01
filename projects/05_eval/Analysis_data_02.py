"""
生成したデータから誤差率や，正解率を算出するプログラム
"""

import os
import json
import numpy as np

def analyze_summary_length(folder_path):
    summary_lengths = []
    content_lengths = []
    all_scores = []
    
    # フォルダ内の全JSONファイルを処理
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                summary_lengths.append(data["target_length"])
                content_lengths.append(data["current_length"])
                id = data["id"]
                all_scores.append(data["total_score"])
            if data["target_length"] != data["current_length"]:
                print(f"{id}")
                
    
    # 誤差計算
    errors = np.abs(np.array(content_lengths) - np.array(summary_lengths))
    max_error = np.max(errors)
    min_error = np.min(errors)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    avg_score = sum(all_scores) / len(all_scores)
    
    # 正解率計算
    exact_match = (np.sum(errors == 0) / len(errors)) * 100
    within_ten_percent = (np.sum(np.abs(errors) <= np.array(summary_lengths) * 0.1) / len(errors)) * 100
    # 各データごとの誤差率（%）を計算
    percentage_errors = (errors / np.array(summary_lengths)) * 100
    # その平均をとる（これが MAPE）
    mean_percentage_error = np.mean(percentage_errors)

    # 手法比較用に、MAPEの元データとtotal_scoreのリストを保存
    output_data = {
        "mape_list": percentage_errors.tolist(),
        "total_scores": all_scores
    }
    with open(f"scores_{os.path.basename(folder_path)}.json", "w") as f:
        json.dump(output_data, f)
    
    # 結果を出力
    results = (
        f"全データ件数: {len(summary_lengths)}\n"
        f"誤差の最大文字数: {int(max_error)}\n"
        f"誤差の最小文字数: {int(min_error)}\n"
        f"平均文字数: {mean_error:.2f}\n"
        f"標準偏差: {std_error:.2f}\n"
        f"平均絶対誤差率 (MAPE): {mean_percentage_error:.2f}%\n"
        f"正解率(ピッタリ): {exact_match:.2f}%\n"
        f"正解率(±10%): {within_ten_percent:.2f}%\n"
        f"平均スコア: {avg_score:.2f}"
    )
    return results

# 使用例
folder_path = "../output_summary/output_zeroshot_FQL_GPT5"  # JSONファイルがあるフォルダのパスに変更
folder_name = os.path.basename(folder_path)
result = analyze_summary_length(folder_path)
print(f"{folder_name}の結果: {result}\n")
# folder_path = "./output_sunao_elyza8B_perfect"  
# result = analyze_summary_length(folder_path)
# print(f"elyza8B: {result}")
