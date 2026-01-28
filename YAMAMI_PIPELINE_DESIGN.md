# yamami Pythonパイプライン設計（ユーザーストーリー付き）

## ユーザーストーリー
1) 研究者は、毎時1枚程度で撮影された大量の定点カメラ写真を持っている。  
2) まず対象写真を選別し、**ファイル名・撮影日時・カメラ種別・解像度**をプロファイルしたい。  
3) 暗さ・霧・雲被覆などで**対象が見えていない状態**を検出し、除外またはマスクしたい。  
4) 風やSDカード交換などで生じた**画角ズレ**を検知し、基準画像へ位置合わせしたい。  
5) 前処理後、**消雪**と**緑葉フェノロジー**を安定的に解析し、図と数値を得たい。  

## パイプライン全体像（更新版）
```
ingest → profile(stats) → prompt_seg(SAM3) → pre_qc → align(imm) → skyline/AOI(参照) → quality_screen → snow → snowmelt → GR → phenology → visualize → export
```

## ステージ設計（入出力）
### 1) ingest（入力収集）
- 入力: 画像ディレクトリ（RAW/JPEG/TIFF混在可、JPEG or TIFFのみ使う）
- 出力: `images/index.csv`（ファイル一覧）
- 主要処理: パス収集、拡張子フィルタ、欠損チェック

### 2) profile（プロファイリング + 画素統計）
- 入力: 画像ファイル
- 出力: `images/profile.csv`
- 主要項目:
  - `filename`, `site`, `camera`, `timestamp`, `width`, `height`, `filesize`
  - `exif_*`（可能なら）
  - `parse_status`（命名規約不一致の検知）
  - 画素統計（暗すぎ判定に使う）  
    `mean_luma`, `std_luma`, `p05_luma`, `p50_luma`, `p95_luma`, `sat_ratio`

### 3) prompt_seg（SAM3 テキストプロンプト判別）
- 入力: 画像
- 出力: `seg/masks/*.png`, `seg/labels.csv`
- 方法:
  - SAM3 に**テキストプロンプト**を与えて領域抽出
  - 一般化メソッドとして実装（対象語彙は可変）
  - **解像度は指定可能、デフォルトは 512x512 にリサイズして推論**
  - 推論マスクは**元解像度へリサイズして戻す**
- 主な用途（標準プロンプト）:
  - 空: `sky`
  - 雲: `cloud`, `fog`
  - 雪: `snow`
- 出力例:
  - `mask_sky`, `mask_cloud`, `mask_fog`, `mask_snow`

### 4) pre_qc（簡易QC／変化点・アンカー用）
- 入力: 画像 + プロファイル + SAM3マスク
- 出力: `images/profile_pre.csv`
- 目的:
  - 変化点検出・アンカー選定に使う**画像単位の軽量QC**
  - AOI未定義でも動く（sky/cloud/fog の全体比率で判定）
- 主要指標:
  - 暗すぎ（輝度中央値/分散）
  - fog/cloud の全体被覆率

### 5) align（カメラズレ検知・補正）
- 入力: 画像 + **全期間の参照画像（global_ref）**
- 出力: `align/warp_params.csv`, `align/aligned/`
- 方法（更新）:
  - **全画像でマッチングしない**前提
  - まず**カメラ移動の変化点**を検出し、**区間（セグメント）**に分割
    - **SAM3で空/雲/霧を除外 → 山肌領域のみ**
    - **山肌領域のエッジ画像で位相相関**を計算し、変位量とピーク強度の時系列から変化点を抽出
    - 低信頼の区間のみ、immの高精度マッチで精査
  - 各セグメント内は**代表画像（アンカー）**を選定
    - `profile_pre`の画質スコア上位から選ぶ
  - **参照プール（global_ref + 既に整列済み画像）**から候補を選ぶ
    - 季節・撮影時間が近い
    - 山肌に雲/霧がかかっていない
    - 空の被雲率が参照画像と近い
  - 候補上位N件（デフォ3）でマッチングし、**再投影誤差**で成否判定（デフォ3px）
  - 成功したらそのセグメントを global_ref 座標系に統一し、**参照プールに追加**
  - 失敗セグメントは、参照プールが更新されたら再試行（`max_rounds`まで）
  - 最終的に失敗したセグメントはフォールバック（手動参照指定／未整列扱い）
  - 得られた変換パラメータを**同一セグメント全画像へ一括適用**
  - **AKAZE/SIFT**または**imm**を選択可能（`old/gcp.py`の`image_match`準拠）
  - imm例: `roma`, `loftr`, `sift-lightglue`, `superpoint-lightglue` など
  - 重いimm手法は自動で縮小（例: 640px）し、座標は元解像度へ再スケール
  - 外れ値除去: Essential/Fundamental 行列フィルタ（USAC_MAGSAC対応）
  - GCP分布の均一化: 空間グリッドでの thinning
  - 幾何補正: 平行移動・回転・射影・歪み補正
  - 出力GCP: `align/matches.csv`（CSVで可視確認可能、マッチング結果も作る）
- 備考:
  - 失敗画像は`align_status=fail`
  - 旧実装参考: `old/akaze_image_matcher.py`
  - imm実装参考: `old/gcp.py`
  - 追加出力: `align/segments.csv`（変化点とセグメント定義）

#### `align/segments.csv` の算出手順（概要）
1) **前処理**  
   - SAM3で `mask_sky/cloud/fog` を作成  
   - 山肌領域 = (not sky/cloud/fog)  
   - ※AOIが既にあれば AOI との交差で絞り込む  
2) **エッジ抽出**  
   - 山肌領域内のみでCanny等のエッジ画像を生成  
3) **位相相関の時系列化**  
   - 連続フレームで位相相関を計算  
   - 変位量（dx, dy）とピーク強度を記録  
4) **変化点検出**  
   - `shift_px_max`超過 or `peak_min`未満が継続する時点を変化点候補  
   - `min_interval_hours` で密な分割を抑制  
5) **セグメント定義**  
   - 変化点で区間分割し `segment_id` を付与  
6) **アンカー選定**  
   - 各区間で `profile_pre` の画質スコア上位を `anchor_path` に選定  
7) **参照候補選定**  
   - 参照プール（global_ref + 整列済み画像）から候補上位N件を抽出  
   - 季節・撮影時間の近さ／山肌の雲霧なし／空の被雲率の近さでスコア化  
8) **マッチング判定**  
   - 候補ごとにimmマッチを実行し、再投影誤差 ≤ 3px なら成功  
   - 成功したセグメントは参照プールへ追加  
9) **反復整列**  
   - 未整列セグメントに対し、参照プール更新後に再試行（`max_rounds`まで）  
10) **global_ref への統一**  
   - 成功したセグメントは `H_to_global` を保存し、一括適用  

### 6) skyline/AOI（稜線抽出・解析対象領域）
- 入力: 補正済み画像
- 出力: `aoi/skyline_*.csv`, `aoi/aoi_mask_*.png`
- 方法:
  - Blue値急変点の走査（既存zsのロジック互換）
  - AOIマスク生成（空領域除外）
- 備考:
  - **稜線より下（山肌側）を解析対象領域として確定**するため、参照稜線を先に決める
  - 参照稜線は「晴天の代表画像（ユーザーが選択）」から作成する想定
  - 参照AOIはセグメントごとの変換でワープして使う

### 7) quality_screen（視認性/品質判定）
- 入力: 画像 + プロファイル + 参照AOI + SAM3マスク
- 出力: `images/profile_qc.csv`, `qc/masks/`
- QCの考え方:
  - **画像単位QC**: 解析に使わない画像を判定
    - 暗すぎ（輝度中央値/分散）
    - 霧・薄雲（`fog`マスク面積）
    - 部分雲被覆（`cloud`マスク面積）
    - **解析対象領域（稜線より下）の雲被覆率**（AOIと`cloud`の交差）
  - **ピクセル単位QC**: 解析に使わない領域をマスク
    - `mask_cloud`, `mask_fog`, `mask_sky` 等
- 統合:
  - QC結果は**プロファイリングと統合**し、`images/profile_qc.csv`に集約
  - 位置合わせ後に評価する場合は、SAM3マスクも同じ変換でワープする
  - **QCマスクは以降の snow / GR / phenology で必ず除外して計算**

### 8) snow（二値化による雪抽出）
- 入力: 画像 + AOI
- 出力: `snow/extract/*.png`, `snow/thresholds.csv`
- 方法:
  - `T = R+G+B` のヒストグラム
  - Otsu閾値で雪/非雪に分類
  - SAM3の`snow`マスクを補助的に利用（任意）

### 9) snowmelt（消雪日推定）
- 入力: `snow/extract` + AOI
- 出力: `snowmelt/doy_map.png`, `snowmelt/doy_map.csv`
- 方法:
  - 期間1/2/3の逐次更新（既存zs互換）
  - 夏期の誤判別補正（反射率の高い岩・ガレ）

### 10) GR（日最大GR生成）
- 入力: 画像 + AOI
- 出力: `gr/daily_max/*.png`, `gr/timeseries.csv`
- 方法:
  - `GR = G/(R+G+B)`
  - 日最大GRを採用（霧・影の影響低減）

### 11) phenology（緑葉開始/終了）
- 入力: GR時系列
- 出力: `phenology/gup.png`, `phenology/gdown.png`, `phenology/gp.png`
- 方法:
  - シグモイド/二重シグモイド近似
  - 変曲点から開始・終了日を推定

### 12) visualize（可視化）
- 入力: 各マップ
- 出力: `viz/*.png`
- 方法:
  - DOYに基づく色分け、凡例バー

### 13) export（成果物整理）
- 入力: 解析結果
- 出力: `export/` 配下にCSV/PNG/TIF
- 付随: 処理ログ・パラメータのスナップショット

## 推奨ディレクトリ構成
```
project/
  data/
    raw/
    processed/
  outputs/
    images/
    seg/
    qc/
    align/
    aoi/
    snow/
    snowmelt/
    gr/
    phenology/
    viz/
    export/
  configs/
    pipeline.yaml
  logs/
```

## 出力形式の方針
- 画像: **PNGまたはTIFF**
- テーブル: **CSV**
- 解析工程の中間生成物も上記に統一（Parquet等は使用しない）

## コンフィグ（例: configs/pipeline.yaml）
```
site: MRD
camera: NikL
image_pattern: "{site}_{azimuth}_{camera}_{band}_{yyyymmdd}_{hhmm}.jpg"
quality:
  brightness_min: 0.08
  contrast_min: 0.05
  fog_std_min: 0.20
  cloud_max: 0.40
  fog_max: 0.30
  aoi_cloud_max: 0.20
pre_qc:
  brightness_min: 0.08
  fog_max: 0.30
  cloud_max: 0.40
sam3:
  prompts:
    - sky
    - cloud
    - fog
    - snow
  resize: 512
align:
  global_ref: "path/to/reference.jpg"  # ユーザーが選ぶ全期間の基準画像
  method: "roma"           # or "akaze", "sift", "loftr", "sift-lightglue" など
  device: "cpu"            # imm使用時
  resize: 640              # heavyメソッドの自動縮小に合わせる
  outlier_filter: "fundamental"  # or "essential", "none"
  ransac_method: "USAC_MAGSAC"
  spatial_thin_grid: 100
  spatial_thin_selection: "center"
  max_rotation_deg: 2.0
  match_candidates: 3
  reproj_err_max: 3.0
  max_rounds: 3
  change_point:
    metric: "phase_corr_edge"  # マスク済みエッジの位相相関
    shift_px_max: 5
    peak_min: 0.2
    min_interval_hours: 6
  anchor_select:
    strategy: "best_qc"    # 画質スコア上位をアンカーに
    window_days: 3
  ref_select:
    season_weight: 1.0
    time_weight: 1.0
    cloud_sky_ratio_weight: 1.0
snow:
  otsu: true
snowmelt:
  stage1:
    start: "05-01"
    end: "05-10"
  stage2:
    checkpoints: ["05-21","05-27","06-02","06-21","06-24","06-28","07-04"]
  stage3:
    start: "07-12"
    end: "08-09"
gr:
  daily_max: true
phenology:
  model: double_sigmoid
```

## データモデル（主要テーブル）
### `profile.csv`
- `path`, `timestamp`, `site`, `camera`, `width`, `height`, `filesize`, `exif_ok`
- `mean_luma`, `std_luma`, `p05_luma`, `p50_luma`, `p95_luma`, `sat_ratio`

### `seg/labels.csv`
- `path`, `mask_sky_area`, `mask_cloud_area`, `mask_fog_area`, `mask_snow_area`

### `images/profile_pre.csv`
- `profile.csv` の全項目
- `is_usable_pre`, `reason_codes_pre[]`, `brightness`, `fog_score`, `cloud_score`

### `images/profile_qc.csv`
- `profile.csv` の全項目
- `is_usable`, `reason_codes[]`, `brightness`, `contrast`, `fog_score`, `cloud_score`
- `cloud_aoi_ratio`（解析対象領域＝稜線より下の雲被覆率）

### `align/warp_params.csv`
- `path`, `dx`, `dy`, `rotation_deg`, `scale`, `status`, `method`

### `align/matches.csv`
- `path`, `u_org`, `v_org`, `u_sim`, `v_sim`
- 由来: `old/gcp.py` の `image_match` 出力（GCP候補）

### `align/segments.csv`
- `segment_id`, `start_ts`, `end_ts`, `anchor_path`, `ref_path`, `status`
- 追加列（推奨）: `dx_med`, `dy_med`, `peak_med`, `n_frames`
- 追加列（推奨）: `global_ref_path`, `H_to_global`, `reproj_err`, `round`, `ref_rank`

## CLI案（例）
```
yamami ingest  data/raw
yamami profile data/raw
yamami prompt-seg --config configs/pipeline.yaml
yamami pre-qc  --config configs/pipeline.yaml
yamami align   --ref outputs/images/reference.jpg
yamami aoi     --config configs/pipeline.yaml
yamami qc      --config configs/pipeline.yaml
yamami snow    --config configs/pipeline.yaml
yamami snowmelt --config configs/pipeline.yaml
yamami gr      --config configs/pipeline.yaml
yamami phenology --config configs/pipeline.yaml
yamami viz     --config configs/pipeline.yaml
yamami export  --config configs/pipeline.yaml
```

## 未実装・要設計ポイント
- SAM3プロンプト辞書の標準化（空/雲/霧/雪以外への拡張）
- ズレ補正の失敗時フォールバック（手動基準更新など）
- サイトごとのパラメータ差分管理（YAML分割）
- 画像出力の標準化（PNG/TIFFの使い分け）

---
この設計を基に、次は「各ステージのPython関数・入出力スキーマの確定」と
「最小実装（MVP）」に落とし込みます。
