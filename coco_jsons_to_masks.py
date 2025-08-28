
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
annotation_json/ にある COCO(coco-annotator) 形式の JSON から、
images/ の各画像に対応する白黒マスク(=二値) またはインデックスマスクを
output_masks/ に出力するスクリプト（パスはコードに固定）。

- 二値マスク: 背景=0(黒), アノテーション領域=255(白)
- インデックス: 背景=0, ピクセル値=カテゴリID (16bit PNGで保存)

必要: pycocotools, opencv-python, numpy, tqdm
  pip install pycocotools opencv-python numpy tqdm
"""

import os
import glob
import sys
import traceback
from typing import Dict

import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO


# =========================
# ★ ここをあなたの環境に合わせて変更 ★
# =========================
ANN_DIR     = "./annotation_json"   # COCO JSON が入ったフォルダ
IMAGES_DIR  = "./images"            # 元画像フォルダ（なくてもwidth/heightがJSONにあれば動く）
OUT_DIR     = "./output_masks"      # 出力マスクフォルダ
MODE        = "binary"              # "binary" or "index"
OVERWRITE   = True                  # 既存PNGを上書きするか
VERBOSE     = True                  # 進捗や情報を出すか
# =========================

import numpy as np
import cv2
from tqdm import tqdm
from pycocotools.coco import COCO

# --- ここから追加 ---
def polygons_to_mask_evenodd(height, width, polygons):
    """
    COCO polygon segmentation (list of [x1,y1,x2,y2,...]) の配列を
    even-odd（XOR合成）で塗って穴を抜く。
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        poly_mask = np.zeros_like(mask)
        cv2.fillPoly(poly_mask, [pts.astype(np.int32)], 1)
        # XORで合成（奇数回重なったところだけ1になる）
        mask = cv2.bitwise_xor(mask, poly_mask)
    return mask

def ann_to_mask_evenodd(coco: COCO, ann: dict, h: int, w: int) -> np.ndarray:
    """
    ann['segmentation'] が polygon のときは even-odd（XOR）で穴を表現。
    RLE（iscrowd 等）の場合は従来通り pycocotools に任せる。
    """
    segm = ann.get("segmentation", None)
    if isinstance(segm, list):
        return polygons_to_mask_evenodd(h, w, segm).astype(np.uint8)
    else:
        # RLEは annToMask を使う（XORの概念なし＝Union）
        return coco.annToMask(ann).astype(np.uint8)
# --- ここまで追加 ---









def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_image_table(images_dir: str) -> Dict[str, str]:
    """画像ファイル名(拡張子込み) -> フルパス の対応表を作る"""
    table: Dict[str, str] = {}
    if not os.path.isdir(images_dir):
        return table
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff", "*.webp")
    for ext in exts:
        for p in glob.glob(os.path.join(images_dir, ext)):
            table[os.path.basename(p)] = p
    return table





def make_mask_for_image(coco: COCO, img_info: dict, mode: str = "binary") -> np.ndarray:
    """1枚の画像に対するマスクを作成"""
    h, w = int(img_info["height"]), int(img_info["width"])
    if mode == "binary":
        out = np.zeros((h, w), dtype=np.uint8)
    elif mode == "index":
        out = np.zeros((h, w), dtype=np.uint16)
    else:
        raise ValueError("MODE must be 'binary' or 'index'.")

    img_id = img_info["id"]
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)

    # 重なりは「後勝ち」
    for ann in anns:
        if ann.get("iscrowd", 0) == 1 and "segmentation" not in ann:
            continue
        # m = coco.annToMask(ann)  # 0/1   ←← ここを差し替える
        m = ann_to_mask_evenodd(coco, ann, h, w)  # 0/1（穴は抜ける）
        if mode == "binary":
            out[m == 1] = 255
        else:
            out[m == 1] = int(ann["category_id"])
    return out
# def make_mask_for_image(coco: COCO, img_info: dict, mode: str = "binary") -> np.ndarray:
#     """1枚の画像に対するマスクを作成"""
#     h, w = int(img_info["height"]), int(img_info["width"])
#     if mode == "binary":
#         out = np.zeros((h, w), dtype=np.uint8)
#     elif mode == "index":
#         out = np.zeros((h, w), dtype=np.uint16)
#     else:
#         raise ValueError("MODE must be 'binary' or 'index'.")

#     img_id = img_info["id"]
#     ann_ids = coco.getAnnIds(imgIds=[img_id])
#     anns = coco.loadAnns(ann_ids)

#     # 重なりは「後勝ち」
#     for ann in anns:
#         if ann.get("iscrowd", 0) == 1 and "segmentation" not in ann:
#             continue
#         m = coco.annToMask(ann)  # 0/1
#         if mode == "binary":
#             out[m == 1] = 255
#         else:
#             out[m == 1] = int(ann["category_id"])
#     return out


def save_mask(mask: np.ndarray, out_path: str):
    """dtypeに応じてPNG保存（uint8→8bit、uint16→16bit）"""
    ok = cv2.imwrite(out_path, mask)
    if not ok:
        raise IOError(f"Failed to write: {out_path}")


def main():
    # 前提チェック
    if not os.path.isdir(ANN_DIR):
        print(f"[ERROR] ANN_DIR が見つかりません: {ANN_DIR}")
        sys.exit(1)
    if not os.path.isdir(IMAGES_DIR):
        # 画像フォルダが無くても JSONに高さ幅があれば動くが、警告だけ出す
        print(f"[WARN] IMAGES_DIR が見つかりません: {IMAGES_DIR} (JSONのwidth/heightだけで処理します)")

    ensure_dir(OUT_DIR)
    image_name_to_path = build_image_table(IMAGES_DIR)

    json_paths = sorted(glob.glob(os.path.join(ANN_DIR, "*.json")))
    if not json_paths:
        print(f"[ERROR] JSONが見つかりません: {ANN_DIR}")
        sys.exit(1)

    total_imgs = 0
    written = 0
    skipped = 0

    for json_path in json_paths:
        if VERBOSE:
            print(f"[INFO] Load COCO: {os.path.basename(json_path)}")
        coco = COCO(json_path)

        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)
        total_imgs += len(imgs)

        for img_info in tqdm(imgs, desc=os.path.basename(json_path), ncols=80):
            file_name = img_info.get("file_name", f"{img_info['id']:012d}.png")
            stem = os.path.splitext(file_name)[0]
            out_path = os.path.join(OUT_DIR, f"{stem}.png")

            if (not OVERWRITE) and os.path.exists(out_path):
                skipped += 1
                continue

            try:
                mask = make_mask_for_image(coco, img_info, mode=MODE)
                save_mask(mask, out_path)
                written += 1
            except Exception:
                print(f"\n[ERROR] 画像ID {img_info.get('id')} ({file_name}) の処理で失敗:")
                traceback.print_exc()

    print("\n================ RESULT ================")
    print(f"  JSON数         : {len(json_paths)}")
    print(f"  画像総数       : {total_imgs}")
    print(f"  書き出し枚数   : {written}")
    print(f"  スキップ(既存) : {skipped}")
    print(f"  出力フォルダ   : {os.path.abspath(OUT_DIR)}")
    print(f"  モード         : {MODE}")
    print("========================================")


if __name__ == "__main__":
    main()









