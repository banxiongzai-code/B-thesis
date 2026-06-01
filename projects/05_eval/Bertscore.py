"""
BERTスコアを算出するプログラム
参照する要約は，XLSum-jaのペアとなっている参照要約とする
"""

import os
import json
import glob
from bert_score import score
from tqdm import tqdm  # 進捗表示（なくても動作します）

# ディレクトリとファイルパス
summary_dir = "../output_summary/output_zeroshot_FQL_calm22B"
ref_file = "./japanese_XLSum_v2.0/japanese_test.jsonl"

# refs（id → summary）を辞書で読み込む
ref_dict = {}
with open(ref_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        ref_dict[data["id"]] = data["summary"]

# 生成要約（cands）と参照要約（refs）を作る
cands = []
refs = []

# output_summary 以下の JSON ファイルを走査
for path in tqdm(glob.glob(os.path.join(summary_dir, "*.json"))):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        cand_text = data["summary"]
        article_id = os.path.basename(path).split("_")[0]

        if article_id in ref_dict:
            cands.append(cand_text)
            refs.append(ref_dict[article_id])

# BERTScoreを計算（日本語モデルで）
P, R, F1 = score(cands, refs, lang="ja", verbose=True)
f1_list = F1.tolist()

# 統計検定用に保存（例：手法名をファイル名にする）
with open(f"scores_{os.path.basename(summary_dir)}_bertscore.json", "w") as f:
    json.dump(f1_list, f)

print(f"BERTScoreの各スコアを保存しました。件数: {len(f1_list)}")

# F1の平均のみを出力
print("\n FLのbertscore")
print(f"Average BERTScore F1: {F1.mean().item():.4f}")
