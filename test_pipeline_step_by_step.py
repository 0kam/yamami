#!/usr/bin/env python3
"""
YAMAMI Pipeline Step-by-Step Test Script

各ステップの機能を順番にテストし、結果を保存します。
ステップ間の情報伝達はファイル(CSV)を介して行われます。

Usage:
    python test_pipeline_step_by_step.py --step 1      # ingestのみ
    python test_pipeline_step_by_step.py --step 2      # segmentのみ
    python test_pipeline_step_by_step.py --step 3      # pre_qcのみ（稜線検出・シフト検出含む）
    python test_pipeline_step_by_step.py --step 4      # alignのみ
    python test_pipeline_step_by_step.py --step 1-4    # step 1から4まで
    python test_pipeline_step_by_step.py               # 全ステップ実行
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# デフォルト設定
DEFAULT_OUTPUT_DIR = Path("test_outputs/full")
DEFAULT_DATA_DIR = Path("test_data_full")
CSV_FILENAME = "pipeline_result.csv"

# ターゲット画像（稜線マッチングの参照画像）
TARGET_IMAGE = "test_data_full/mrd_085_eos_vis_20200713_0905_R.JPG"


def get_csv_path(output_dir: Path) -> Path:
    """累積結果CSVのパスを取得"""
    return output_dir / CSV_FILENAME


def step1_ingest(data_dir: Path, output_dir: Path) -> None:
    """
    Step 1: 画像ファイルの発見とプロファイル作成

    入力: data_dir (画像ディレクトリ)
    出力: output_dir/pipeline_result.csv
    """
    print("\n" + "=" * 60)
    print("Step 1: ingest - 画像ファイルの発見とプロファイル作成")
    print("=" * 60)

    from yamami.ingest import ingest

    print(f"\n入力ディレクトリ: {data_dir}")
    print("画像をスキャン中...")

    df = ingest(data_dir)

    print(f"\n発見した画像数: {len(df)}")
    print("\nプロファイル結果:")
    display_cols = ["filename", "timestamp", "width", "height", "mean_luma", "std_luma"]
    print(df[display_cols].to_string())

    # 統計サマリー
    print("\n輝度統計サマリー:")
    print(f"  mean_luma: min={df['mean_luma'].min():.1f}, "
          f"max={df['mean_luma'].max():.1f}, "
          f"mean={df['mean_luma'].mean():.1f}")

    # CSVに保存
    csv_path = get_csv_path(output_dir)
    df.to_csv(csv_path, index=False)
    print(f"\n結果を保存: {csv_path}")


def step2_segment(output_dir: Path) -> None:
    """
    Step 2: SAM3によるセマンティックセグメンテーション

    入力: output_dir/pipeline_result.csv (step1の出力)
    出力: output_dir/pipeline_result.csv (segment結果を追加)
           output_dir/segment/ (マスク画像、プレビュー画像)
    """
    print("\n" + "=" * 60)
    print("Step 2: segment - SAM3によるセグメンテーション")
    print("=" * 60)

    from yamami.segment import segment

    # CSVから読み込み
    csv_path = get_csv_path(output_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Step 1の出力が見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\n読み込んだ画像数: {len(df)}")

    print("\nセグメンテーションを実行中...")
    print("(SAM3モデルのロードに時間がかかる場合があります)")

    # 色指定（BGR形式）
    custom_colors = {
        "sun": (0, 255, 255),
        "raindrop": (255, 0, 0),
        "reflection": (0, 255, 0),
        "fog": (200, 200, 200),
        "cloud": (255, 255, 255),
        "sky": (255, 180, 100),
        "mountain": (150, 75, 0),
        "human": (0, 0, 255),
        "vehicle": (255, 0, 255),
        "snow": (255, 255, 200),
    }

    prompts = [
        "human", "vehicle", "sun", "raindrop", "reflection",
        "fog", "cloud", "sky", "snow", "mountain"
    ]

    labels_long, masks_dict = segment(
        df,
        output_dir=output_dir / "segment",
        prompts=prompts,
        colors=custom_colors,
        resize=0,
    )

    # Long format -> Wide format に変換してマージ
    pivot = labels_long.pivot(index="path", columns="prompt", values="ratio").reset_index()
    pivot.columns = ["path"] + [f"{col}_ratio" for col in pivot.columns[1:]]
    df = df.merge(pivot, on="path", how="left")

    # 結果表示
    print(f"\n処理した画像数: {len(df)}")
    ratio_cols = [c for c in df.columns if c.endswith("_ratio")]
    print("\nセグメンテーション結果 (ratio):")
    print(df[["filename"] + ratio_cols].to_string())

    # 統計サマリー
    print("\nプロンプト別統計:")
    for col in ratio_cols:
        print(f"  {col}: mean={df[col].mean():.4f}, max={df[col].max():.4f}")

    # CSVに保存（上書き）
    df.to_csv(csv_path, index=False)
    print(f"\n結果を保存: {csv_path}")


def step3_pre_qc(output_dir: Path, target_image: str = None) -> None:
    """
    Step 3: 事前QC + 稜線ベースのシフト検出 + セグメント定義

    入力: output_dir/pipeline_result.csv (step2の出力)
          output_dir/segment/masks/ (マスクファイル)
    出力: output_dir/pipeline_result.csv (QC結果・シフト結果を追加)
          output_dir/ridgeline/ (稜線可視化画像)
    """
    print("\n" + "=" * 60)
    print("Step 3: pre_qc - 品質管理 + 稜線ベースシフト検出")
    print("=" * 60)

    from yamami.qc import ridge_qc

    # CSVから読み込み
    csv_path = get_csv_path(output_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Step 2の出力が見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\n読み込んだ画像数: {len(df)}")

    # 必要なカラムの確認
    required_cols = ["fog_ratio", "sun_ratio", "reflection_ratio"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Step 2のセグメンテーション結果がありません: {missing}")

    # ターゲット画像の決定
    if target_image is None:
        target_image = TARGET_IMAGE

    target_path = Path(target_image)
    if not target_path.exists():
        for _, row in df.iterrows():
            if target_path.name in row["path"]:
                target_image = row["path"]
                target_path = Path(target_image)
                break

    masks_dir = output_dir / "segment" / "masks"
    if not masks_dir.exists():
        raise FileNotFoundError(f"マスクディレクトリが見つかりません: {masks_dir}")

    print(f"  ターゲット画像: {target_path.name}")

    # --- ridge_qc で一括処理 ---
    print("\n基本QC + 適応的稜線検出 + RANSACマッチング + セグメント検出...")

    df = ridge_qc(
        df,
        masks_dir=masks_dir,
        target_image=target_image,
        output_dir=output_dir,
        luma_min=20.0,
        luma_max=240.0,
        fog_max=0.01,
        sun_max=0.01,
        reflection_max=0.01,
        fog_skip_threshold=0.05,
        displacement_threshold=5.0,
    )

    # --- 結果表示 ---
    print("\n" + "-" * 120)
    print(f"{'filename':<40} {'method':>10} {'b_ratio':>7} {'err_med':>8} "
          f"{'dx':>6} {'dy':>6} {'angle':>6} {'scale':>5} {'match':>6}")
    print("-" * 120)

    for _, row in df.iterrows():
        method = row.get("ridge_method", "")
        br = row.get("ridge_blue_ratio", np.nan)
        em = row.get("ridge_error_median", np.nan)
        dx = row.get("ridge_dx", np.nan)
        dy = row.get("ridge_dy", np.nan)
        angle = row.get("ridge_angle", np.nan)
        scale = row.get("ridge_scale", np.nan)
        success = row.get("ridge_match_success", False)

        br_str = f"{br:.3f}" if not np.isnan(br) else ""
        em_str = f"{em:.1f}" if not (np.isnan(em) or np.isinf(em)) else ""
        dx_str = f"{dx:.1f}" if not np.isnan(dx) else ""
        dy_str = f"{dy:.1f}" if not np.isnan(dy) else ""
        angle_str = f"{angle:.2f}" if not np.isnan(angle) else ""
        scale_str = f"{scale:.3f}" if not np.isnan(scale) else ""
        status = "OK" if success else ""

        print(f"{row['filename']:<40} {method:>10} {br_str:>7} {em_str:>8} "
              f"{dx_str:>6} {dy_str:>6} {angle_str:>6} {scale_str:>5} {status:>6}")

    # --- セグメント情報 ---
    print("\n" + "=" * 60)
    print("セグメント情報")
    print("=" * 60)

    if "segment_id" in df.columns and not df["segment_id"].isna().all():
        for seg_id in sorted(df["segment_id"].dropna().unique()):
            seg_df = df[df["segment_id"] == seg_id]
            n_confirmed = seg_df["ridge_confirmed"].sum() if "ridge_confirmed" in seg_df.columns else 0
            first = seg_df.iloc[0]["filename"]
            last = seg_df.iloc[-1]["filename"]
            dx_med = seg_df["ridge_dx"].median()
            dy_med = seg_df["ridge_dy"].median()
            print(f"  seg {int(seg_id)}: {len(seg_df)}枚 (確認{int(n_confirmed)})  "
                  f"dx={dx_med:+.1f} dy={dy_med:+.1f}")
            print(f"         {first} .. {last}")
    else:
        print("  セグメント情報なし")

    # --- 統計サマリー ---
    print("\n" + "=" * 60)
    print("統計サマリー")
    print("=" * 60)
    print(f"  基本QC usable: {df['is_usable'].sum()}")
    print(f"  稜線マッチング成功: {df['ridge_match_success'].sum()}")
    if "segment_id" in df.columns and not df["segment_id"].isna().all():
        print(f"  セグメント数: {int(df['segment_id'].nunique())}")

    # CSVに保存（上書き）
    df.to_csv(csv_path, index=False)
    print(f"\n結果を保存: {csv_path}")


def step4_align(output_dir: Path, target_image: str = None) -> None:
    """
    Step 4: ラウンド制アライメント + レンズ歪み補正

    入力: output_dir/pipeline_result.csv (step3の出力)
          output_dir/segment/masks/ (マスクファイル)
    出力: output_dir/pipeline_result.csv (align結果を追加)
          output_dir/aligned/ (アライメント済み画像)
          output_dir/aligned/alignment_params.json (パラメータ)
    """
    print("\n" + "=" * 60)
    print("Step 4: align - ラウンド制アライメント + レンズ歪み補正")
    print("=" * 60)

    from yamami.align import align

    # CSVから読み込み
    csv_path = get_csv_path(output_dir)
    if not csv_path.exists():
        raise FileNotFoundError(f"Step 3の出力が見つかりません: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"\n読み込んだ画像数: {len(df)}")

    # 必要なカラムの確認
    required_cols = ["segment_id", "ridge_match_success"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Step 3の稜線解析結果がありません: {missing}")

    # ターゲット画像の決定
    if target_image is None:
        target_image = TARGET_IMAGE

    target_path = Path(target_image)
    if not target_path.exists():
        for _, row in df.iterrows():
            if target_path.name in row["path"]:
                target_image = row["path"]
                target_path = Path(target_image)
                break

    masks_dir = output_dir / "segment" / "masks"
    if not masks_dir.exists():
        raise FileNotFoundError(f"マスクディレクトリが見つかりません: {masks_dir}")

    print(f"  ターゲット画像: {target_path.name}")
    print(f"  マスクディレクトリ: {masks_dir}")

    # --- align 実行 ---
    print("\nアライメント実行中...")
    print("  マッチング手法: minima-roma")
    print("  レンズ歪み補正: あり")
    print("  稜線バリデーション: あり")

    df = align(
        df,
        target_image=target_image,
        masks_dir=str(masks_dir),
        method="minima-roma",
        output_dir=str(output_dir),
        device="cuda",
        max_rounds=5,
        ransac_thresh = 5.0,
        min_ransac_inliers=50,
        max_rmse=5.0,
        ridge_validation=True,
        estimate_distortion="global",
        grid_size=50
    )

    # --- 結果表示 ---
    print("\n" + "-" * 100)
    print(f"{'filename':<40} {'status':>8} {'rmse':>7} {'matches':>8} "
          f"{'anchor':<30} {'round':>5}")
    print("-" * 100)

    for _, row in df.iterrows():
        status = row.get("align_status", "")
        rmse = row.get("align_rmse", np.nan)
        matches = row.get("align_num_matches", 0)
        anchor = row.get("align_anchor", "")
        rnd = row.get("align_round", -1)

        rmse_str = f"{rmse:.2f}" if not np.isnan(rmse) else ""
        matches_str = str(int(matches)) if matches > 0 else ""
        rnd_str = str(int(rnd)) if rnd > 0 else ""

        print(f"{row['filename']:<40} {status:>8} {rmse_str:>7} {matches_str:>8} "
              f"{anchor:<30} {rnd_str:>5}")

    # --- セグメント別サマリー ---
    print("\n" + "=" * 60)
    print("セグメント別アライメント結果")
    print("=" * 60)

    if "segment_id" in df.columns and not df["segment_id"].isna().all():
        for seg_id in sorted(df["segment_id"].dropna().unique()):
            seg_df = df[df["segment_id"] == seg_id]
            n_total = len(seg_df)
            n_success = (seg_df["align_status"] == "success").sum()
            rmse_vals = seg_df["align_rmse"].dropna()
            rmse_med = rmse_vals.median() if len(rmse_vals) > 0 else np.nan
            anchor = seg_df["align_anchor"].iloc[0] if len(seg_df) > 0 else ""
            rnd = seg_df["align_round"].iloc[0] if len(seg_df) > 0 else -1

            rmse_str = f"RMSE={rmse_med:.2f}" if not np.isnan(rmse_med) else "RMSE=N/A"
            rnd_str = f"R{int(rnd)}" if rnd > 0 else "失敗"

            print(f"  seg {int(seg_id)}: {n_success}/{n_total}枚成功  "
                  f"{rmse_str}  {rnd_str}  anchor={anchor}")

    # --- 統計サマリー ---
    print("\n" + "=" * 60)
    print("アライメント統計サマリー")
    print("=" * 60)
    n_success = (df["align_status"] == "success").sum()
    n_failed = (df["align_status"] == "failed").sum()
    print(f"  成功: {n_success}")
    print(f"  失敗: {n_failed}")
    if n_success > 0:
        rmse_vals = df.loc[df["align_status"] == "success", "align_rmse"]
        print(f"  RMSE (中央値): {rmse_vals.median():.2f} px")
        print(f"  RMSE (最大): {rmse_vals.max():.2f} px")

    # 出力ファイル確認
    aligned_dir = output_dir / "aligned"
    if aligned_dir.exists():
        aligned_files = list(aligned_dir.glob("*.JPG")) + list(aligned_dir.glob("*.jpg"))
        print(f"\n  アライメント済み画像: {len(aligned_files)}枚")
        params_path = aligned_dir / "alignment_params.json"
        if params_path.exists():
            print(f"  パラメータファイル: {params_path}")

    # CSVに保存（上書き）
    df.to_csv(csv_path, index=False)
    print(f"\n結果を保存: {csv_path}")


def parse_step_range(step_arg: str) -> list[int]:
    """ステップ引数をパース（例: "1", "1-3", "2-4"）"""
    if "-" in step_arg:
        start, end = step_arg.split("-")
        return list(range(int(start), int(end) + 1))
    else:
        return [int(step_arg)]


def main():
    parser = argparse.ArgumentParser(
        description="YAMAMI Pipeline Step-by-Step Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_pipeline_step_by_step.py --step 1           # ingestのみ
    python test_pipeline_step_by_step.py --step 2           # segmentのみ
    python test_pipeline_step_by_step.py --step 3           # pre_qc + 稜線検出
    python test_pipeline_step_by_step.py --step 4           # align
    python test_pipeline_step_by_step.py --step 1-4         # step 1から4まで
    python test_pipeline_step_by_step.py                    # 全ステップ
    python test_pipeline_step_by_step.py --data test_data_summer  # 別のデータ
        """
    )
    parser.add_argument(
        "--step", "-s",
        default="1-4",
        help="実行するステップ（例: 1, 2, 1-4）"
    )
    parser.add_argument(
        "--data", "-d",
        default=str(DEFAULT_DATA_DIR),
        help=f"入力データディレクトリ（デフォルト: {DEFAULT_DATA_DIR}）"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"出力ディレクトリ（デフォルト: {DEFAULT_OUTPUT_DIR}）"
    )
    parser.add_argument(
        "--target-image",
        default=TARGET_IMAGE,
        help=f"稜線マッチングのターゲット画像（デフォルト: {TARGET_IMAGE}）"
    )

    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    steps = parse_step_range(args.step)

    print("=" * 60)
    print("YAMAMI Pipeline Step-by-Step Test")
    print("=" * 60)
    print(f"データディレクトリ: {data_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"実行ステップ: {steps}")

    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 各ステップを実行
    if 1 in steps:
        step1_ingest(data_dir, output_dir)

    if 2 in steps:
        step2_segment(output_dir)

    if 3 in steps:
        step3_pre_qc(output_dir, args.target_image)

    if 4 in steps:
        step4_align(output_dir, args.target_image)

    # 完了メッセージ
    print("\n" + "=" * 60)
    print(f"Step {steps[0]}-{steps[-1]} 完了" if len(steps) > 1 else f"Step {steps[0]} 完了")
    print("=" * 60)

    # 結果サマリー（CSVが存在する場合）
    csv_path = get_csv_path(output_dir)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\n総画像数: {len(df)}")
        if "is_usable" in df.columns:
            print(f"基本QC usable: {df['is_usable'].sum()}")
        if "ridge_match_success" in df.columns:
            print(f"稜線マッチング成功: {df['ridge_match_success'].sum()}")
        if "segment_id" in df.columns and not df["segment_id"].isna().all():
            print(f"セグメント数: {df['segment_id'].nunique()}")
        if "align_status" in df.columns:
            n_aligned = (df["align_status"] == "success").sum()
            print(f"アライメント成功: {n_aligned}")
        print(f"\n結果: {csv_path}")


if __name__ == "__main__":
    main()
