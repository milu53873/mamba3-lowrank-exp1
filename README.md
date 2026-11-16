# mamba3-lowrank-exp1

自己從零訓練一個約 17M 參數的小 Mamba-3，把每層 mixer 的 `in_proj`（z、x 部分）與 `out_proj` 換成兩個小矩陣（低秩 r=64），和沒改的 baseline 比較。
Train a ~17M-parameter Mamba-3 from scratch on WikiText-2 and compare the dense baseline with a low-rank (r=64) mixer.

## 結果（3,000 步，單一種子，WikiText-2 validation）

| 模型 | 參數 | 驗證 PPL | 秒／步 |
|---|---|---|---|
| baseline | 17.02M | 167.68 | 4.26 |
| 低秩 r=64 | 15.71M | 167.10 | 3.63 |

- 區塊乘法量 603,530 → 341,386（−43.4%，手算）；整個模型每 token 乘法 −7.3%（LM head 占 72%）。
- 兩者 PPL 差距在單次實驗的雜訊範圍內，只能說沒有變差。
- 乘法量是手算，沒有量測硬體。

## 檔案

| 檔案 | 內容 |
|---|---|
| `train.py` | 訓練腳本（`--variant baseline｜lowrank`、`--rank`、`--compare`） |
| `mamba3_baseline.py` | baseline 區塊（純 PyTorch，原版不改） |
| `mamba3_lowrank.py` | 低秩區塊（新增 `LowRankLinear`，`lowrank_r` 參數） |
| `param_breakdown.py` | 算參數與每 token 乘法分布 |
| `data/wikitext-2-raw-v1.zip` | WikiText-2（沒放的話 `train.py` 會自己下載） |
| `runs/` | 本次實驗的 json、log |

不需要安裝 `mamba_ssm`（`train.py` 找不到時會用空的 `tap` 取代）。需要 `torch`、`transformers`、`einops`。

## 執行

```bash
# 小測（不用下載，CPU 可）
python train.py --synthetic --variant lowrank --rank 8 --steps 5 --d_model 64 --n_layer 1 --seq 32 --bs 2

# 完整實驗：一定要加 --tokenizer gpt2（預設是需授權的 Llama-3.1）
mkdir -p runs
python -u train.py --tokenizer gpt2 --variant baseline --bs 16 --steps 3000 --eval_every 250 --out runs/gpt2_baseline.json > runs/gpt2_baseline.log 2>&1
python -u train.py --tokenizer gpt2 --variant lowrank --rank 64 --bs 16 --steps 3000 --eval_every 250 --out runs/gpt2_lowrank_r64.json > runs/gpt2_lowrank_r64.log 2>&1

# 比較、參數分布
python train.py --compare runs/gpt2_baseline.json runs/gpt2_lowrank_r64.json
python param_breakdown.py
```

RTX 6000（24 GB）上 batch 16 約需 10.8 GB；每步約 4.3 秒，單次約 3 到 3.6 小時（由步數乘每步秒數推算）。
上面的指令是照程式參數與結果 json 重建的，不是當時指令的逐字複製。

## 限制

- 單一種子、單一 rank，沒有掃 rank；PPL 偏高（資料只有 2.4M token、有過擬合），只有同設定下的比較有意義。
- 量測是 WikiText-2 validation（最多 50 個 batch），GPT-2 tokenizer，不能和其他 tokenizer 的 perplexity 比。

## 來源

`mamba3_baseline.py`、`mamba3_lowrank.py` 改自 Dao AI Lab 的 Mamba（state-spaces/mamba），原始授權與版權以上游專案為準。
