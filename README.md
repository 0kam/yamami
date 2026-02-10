# YAMAMI

山岳フェノロジー解析パイプライン - 定点カメラで撮影されたタイムラプス画像から、積雪消長と植生緑化のタイミングを抽出するPythonライブラリです。

## 概要

YAMAMIは、山岳地帯の定点観測カメラ画像を解析し、以下の情報を自動抽出するパイプラインです：

- **融雪時期（Snowmelt DOY）**: ピクセル単位での融雪日の推定
- **緑化開始日（Green-up DOY）**: 植生の緑化が始まる日
- **緑化終了日（Green-down DOY）**: 植生の緑化が終わる日
- **最大緑化日（Green-max DOY）**: 緑度が最大となる日

## パイプライン構成

処理は以下の順序で実行されます：

```
ingest → segment → pre_qc/ridge_qc → align → aoi → gr/phenology → snow/snowmelt → viz → export
```

### 各モジュールの概要

| モジュール | 機能 | 入力 | 出力 |
|-----------|------|------|------|
| **ingest** | 画像ファイルの探索・メタデータ・輝度統計の抽出 | ディレクトリパス | profile DataFrame |
| **segment** | SAM3によるセグメンテーション | profile DataFrame | labels(long format), masks/ |
| **pre_qc** | 基本QC（輝度・霧・太陽・反射による閾値判定） | profile DataFrame（segment結果マージ済み） | profile + is_usable, reason_codes |
| **ridge_qc** | 稜線ベースQC + カメラシフト検出 + セグメント定義 | profile DataFrame, masks/, target画像 | profile + ridge_*, segment_id |
| **align** | ラウンド制アライメント + レンズ歪み補正 | profile DataFrame, target画像, masks/ | profile + align_*, aligned/ |
| **aoi** | 解析対象領域（AOI）の抽出 | 基準画像 | skyline.csv, aoi_mask |
| **gr** | 緑度比の計算 | 位置合わせ済み画像, aoi_mask | gr_timeseries, gr_images/ |
| **phenology** | フェノロジー曲線フィッティング | gr_timeseries | phenology DataFrame |
| **snow** | 積雪検出 | profile DataFrame, aoi_mask | snow_masks/, thresholds |
| **snowmelt** | 融雪日の推定 | snow_masks/, aoi_mask | snowmelt DataFrame |
| **viz** | DOYマップの可視化 | phenology/snowmelt DataFrame | gup.png, doy_map.png等 |
| **export** | 結果のエクスポート | 全中間出力 | export/, processing_log.json |

## 主要機能の詳細

### 1. 画像取り込み（ingest）

`ingest()` は画像ファイルの探索・メタデータ抽出・輝度統計計算を一括で行います：

```python
from yamami import ingest

# 画像ファイルの探索とプロファイル作成
df = ingest(directory="./images", patterns=["*.jpg", "*.tiff"])
```

**出力DataFrameのカラム:**
- `path`, `filename`: ファイル情報
- `timestamp`: ファイル名から抽出したタイムスタンプ
- `width`, `height`, `filesize`: 画像サイズ
- `mean_luma`, `std_luma`, `p05_luma`, `p50_luma`, `p95_luma`: 輝度統計

### 2. セグメンテーション（segment）

Meta SAM3を使用したセマンティックセグメンテーション:

```python
from yamami import segment

labels_long, masks_dict = segment(
    profile=df,
    prompts=["fog", "cloud", "sky", "sun", "reflection", "snow", "mountain"],
    resize=512,
    device="auto"  # "cpu", "cuda", または "auto"
)
```

**戻り値:**
- `labels_long`: Long format DataFrame（path, prompt, ratio）
- `masks_dict`: filepath -> {prompt -> bool mask} の辞書

Wide formatへの変換例：
```python
pivot = labels_long.pivot(index="path", columns="prompt", values="ratio").reset_index()
pivot.columns = ["path"] + [f"{col}_ratio" for col in pivot.columns[1:]]
df = df.merge(pivot, on="path", how="left")
```

### 3. 品質管理（qc）

#### 基本QC（pre_qc）

輝度・セグメンテーション結果に基づく自動フィルタリング:

```python
from yamami import pre_qc

df = pre_qc(
    df,
    luma_min=20.0,
    luma_max=240.0,
    fog_max=0.01,
    sun_max=0.01,
    reflection_max=0.01,
)
```

**出力に追加されるカラム:**
- `is_usable`: 使用可否（True/False）
- `reason_codes`: 除外理由（例: "TOO_DARK,HIGH_FOG"）

> **Note**: `pre_qc()` は DataFrame 内のカラム（`mean_luma`, `fog_ratio`, `sun_ratio`, `reflection_ratio`）を直接参照します。事前に segment 結果を merge しておく必要があります。

#### 稜線ベースQC（ridge_qc）

稜線検出・RANSACマッチング・カメラシフト検出・セグメント定義を一括で実行:

```python
from yamami import ridge_qc

df = ridge_qc(
    df,
    masks_dir="./output/segment/masks",
    target_image="reference.jpg",
    output_dir="./output",
    luma_min=20.0,
    luma_max=240.0,
    fog_max=0.01,
    sun_max=0.01,
    reflection_max=0.01,
    fog_skip_threshold=0.05,
    displacement_threshold=5.0,
)
```

**出力に追加されるカラム:**
- `is_usable`, `reason_codes`: 基本QCの結果
- `ridge_method`: 使用した稜線検出手法（"sky_index" or "blue_gradient"）
- `ridge_dx`, `ridge_dy`, `ridge_angle`, `ridge_scale`: カメラシフトパラメータ
- `ridge_match_success`: 稜線マッチング成功/失敗
- `segment_id`: カメラシフトセグメントID
- `ridge_confirmed`: 稜線で確認済みか

### 4. 稜線検出（ridgeline）

稜線検出・マッチングの低レベルAPI:

```python
from yamami import (
    detect_ridgeline, match_ridgelines_ransac, detect_segments,
    compute_blue_ratio, get_sam_ridge, load_mask
)

# SAMマスクから稜線を取得
sam_ridge = get_sam_ridge(mask_path)

# 稜線検出（Sky IndexまたはBlue Gradientを自動選択）
ridgeline, method, blue_ratio = detect_ridgeline(
    img, sam_ridge, sky_mask, cloud_mask,
    blue_ratio_threshold=0.34
)

# 稜線のRANSACマッチング
result = match_ridgelines_ransac(ridge_target, ridge_query)
# result: {dx, dy, angle, scale, error_median, match_success, residual_mask}
```

### 5. 位置合わせ（align）

ラウンド制アライメントとレンズ歪み補正:

```python
from yamami import align

df = align(
    df,
    target_image="reference.jpg",
    masks_dir="./output/segment/masks",
    method="superpoint-lightglue",
    output_dir="./output",
    device="cuda",
    max_rounds=5,
    min_ransac_inliers=20,
    max_rmse=5.0,
    ridge_validation=True,
    estimate_distortion="global",  # "global", "per_segment", "off"
    grid_size=100,
    ransac_thresh=10.0,
)
```

**出力に追加されるカラム:**
- `align_status`: "success" or "failed"
- `align_rmse`: 再投影RMSE (px)
- `align_num_matches`: マッチング数
- `align_anchor`: 使用したアンカー画像
- `align_round`: アライメントが成功したラウンド番号
- `align_method`: 使用したマッチング手法

**出力ファイル:**
- `aligned/`: アライメント済み画像
- `aligned/alignment_params.json`: アライメントパラメータ

**対応する特徴点マッチング手法:**
- OpenCV: `akaze`, `sift`
- IMM（Image Matching Models）: `roma`, `loftr`, `sift-lightglue`, `superpoint-lightglue`, `tiny-roma`, `minima-roma` 等

### 6. AOI抽出（aoi）

青チャンネルの勾配を使用したスカイライン検出:

```python
from yamami import aoi

skyline, aoi_mask = aoi(ref_image="reference.jpg", output_dir="./output")
```

**出力:**
- `skyline`: DataFrame（x, y座標）
- `aoi_mask`: バイナリマスク（ndarray）

### 7. 緑度比計算（gr）

```python
from yamami import gr

daily_images, timeseries = gr(
    profile=df,
    aoi_mask=aoi_mask,
    daily_max=True,
    output_dir="./output"
)
```

**緑度比（GR）の計算式:**
```
GR = G / (R + G + B)
```

### 8. フェノロジー解析（phenology）

ダブルシグモイドモデルによる曲線フィッティング:

```python
from yamami import phenology

pheno = phenology(
    gr_data=timeseries,
    model="double_sigmoid",
    output_dir="./output"
)
```

**モデル式:**
```
f(t) = base + amp1/(1+exp(-k1*(t-t1))) - amp2/(1+exp(-k2*(t-t2)))
```

**phenology DataFrame の出力カラム:**
- `row`, `col`: ピクセル座標
- `gup_doy`: 緑化開始日（t1の変曲点）
- `gdown_doy`: 緑化終了日（t2の変曲点）
- `gmax_doy`: 最大緑度日
- `gmax_value`: 最大緑度値
- `fit_rmse`: フィッティング誤差
- `fit_status`: フィッティング状態

### 9. 積雪解析（snow/snowmelt）

```python
from yamami import snow, snowmelt

# 積雪検出（大津法による自動閾値決定）
snow_masks, thresholds = snow(
    profile=df,
    aoi_mask=aoi_mask,
    otsu=True,
    output_dir="./output"
)

# 融雪日の推定
melt = snowmelt(
    snow_masks=snow_masks,
    aoi_mask=aoi_mask,
    output_dir="./output"
)
```

**snowmelt DataFrame の出力カラム:**
- `row`, `col`: ピクセル座標
- `doy`: 融雪日（Day of Year）
- `confidence`: 信頼度
- `status`: 状態（"melted", "no_snow", "persistent_snow", "uncertain"）

### 10. 可視化（viz）

DOYをカラーマップで可視化:

```python
from yamami import viz

viz(
    data=pheno,  # または snowmelt
    colormap="viridis",
    vmin=100,
    vmax=250,
    output_dir="./output"
)
```

**出力画像:**
- `gup.png`: 緑化開始日マップ
- `gdown.png`: 緑化終了日マップ
- `gmax.png`: 最大緑度日マップ
- `doy_map.png`: 融雪日マップ

### 11. エクスポート（export）

結果の整理と来歴（Provenance）の記録:

```python
from yamami import export

export(
    output_dir="./output",
    export_dir="./export",
    include_log=True
)
```

**processing_log.json の内容:**
- 処理日時
- パイプラインバージョン
- 全出力ファイルのSHA256ハッシュ
- 使用パラメータ

## 入出力フォーマット

### 入力
- **画像**: JPEG (.jpg, .jpeg), TIFF (.tif, .tiff)
- **タイムスタンプ**: ファイル名の正規表現パース、またはEXIFメタデータ

### 出力
- **表形式データ**: CSV（ヘッダ付き）
- **画像**: PNG（マスク、可視化）
- **数値配列**: NumPy .npz（マスク中間データ）
- **メタデータ**: JSON

## 依存ライブラリ

### コア（常に必要）
- numpy, pandas - データ操作
- opencv-python - 画像I/O・処理
- scipy - 曲線フィッティング
- pillow - EXIF抽出
- tqdm - プログレスバー
- matplotlib - 可視化

### セグメンテーション（実質必須）
- **torch** - PyTorch 2.7+（CUDA 12.6対応）
- **sam3** - Meta Segment Anything Model 3
  - `segment`モジュールに必須
  - SAM3がないとlabels（fog_ratio, cloud_ratio等）が生成されず、QCが正しく機能しない
  - CUDAが必須（CPUでは動作しない）

#### SAM3のセットアップ手順

SAM3はyamamiの依存関係として自動インストールされますが、**モデルチェックポイントのダウンロードにはHugging Faceでの事前承認が必要**です。

**Step 1: Hugging Faceでアクセスをリクエスト**

1. [SAM3 Hugging Faceリポジトリ](https://huggingface.co/facebook/sam3)にアクセス
2. 「Request access」ボタンをクリック
3. 利用目的等を入力して申請
4. 承認メールが届くまで待機（通常数時間〜数日）

**Step 2: Hugging Face認証（承認後）**

```bash
# huggingface_hubのインストール（未インストールの場合）
pip install huggingface_hub

# ログイン（アクセストークンが必要）
hf auth login
# プロンプトに従いアクセストークンを入力
# トークンは https://huggingface.co/settings/tokens で生成
```

**Step 3: 動作確認**

```python
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# モデルのロード（初回は自動ダウンロード）
model = build_sam3_image_model()
processor = Sam3Processor(model)

# テスト画像でセグメンテーション
image = Image.open("test.jpg")
inference_state = processor.set_image(image)
output = processor.set_text_prompt(state=inference_state, prompt="sky")
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
print(f"検出数: {len(masks)}")
```

#### SAM3の動作要件

| 項目 | 要件 |
|------|----------|
| Python | 3.12 |
| PyTorch | 2.7以上 |
| CUDA | 12.6以上 |
| GPU VRAM | 8GB以上推奨 |

> **Note**: PyTorchはyamamiインストール時に自動インストールされますが、CUDA対応版が必要な場合は事前に手動インストールしてください：
> ```bash
> pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
> ```

#### トラブルシューティング

- **「Access denied」エラー**: Hugging Faceでのアクセス承認が完了していないか、`hf auth login`が実行されていません
- **CUDA関連エラー**: `nvidia-smi`でGPUが認識されているか確認してください
- **メモリ不足**: `segment`モジュールの`resize`パラメータを小さくして画像サイズを縮小してください

### 位置合わせ（手法により必要）
- **imm** (image-matching-models) - 高度な特徴点マッチング
  - `roma`, `loftr`, `sift-lightglue`, `superpoint-lightglue`等を使用する場合に必須
  - `akaze`, `sift`を使用する場合はOpenCVのみで動作（immは不要）

## インストール

```bash
pip install yamami
```

## 動作要件

- Python 3.12
- GPU（CUDA 12.6以上）必須（SAM3によるセグメンテーションに必要）

## ライセンス

MIT License
