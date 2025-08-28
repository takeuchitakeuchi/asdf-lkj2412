

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob
from pathlib import Path
import numpy as np
import cv2
from pycocotools.coco import COCO

# === あなたのパス ===
ANN_DIR    = "./annotation_json"
IMAGES_DIR = "./images"
OUT_DIR    = "./output_masks"
DBG_DIR    = "./debug_overlays"
TOPK       = 3  # 面積トップK注釈を可視化

os.makedirs(DBG_DIR, exist_ok=True)

def overlay_mask(img, mask, alpha=0.5):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(img)
    color[..., 1] = 255  # 緑
    m3 = np.stack([mask]*3, axis=-1).astype(bool)
    out = img.copy()
    out[m3] = (alpha*color[m3] + (1-alpha)*img[m3]).astype(np.uint8)
    return out

def main():
    jsons = sorted(glob.glob(os.path.join(ANN_DIR, "*.json")))
    if not jsons:
        print("[ERR] JSONなし")
        return

    for jp in jsons:
        print(f"\n[INFO] {os.path.basename(jp)}")
        coco = COCO(jp)
        img_ids = coco.getImgIds()
        imgs = coco.loadImgs(img_ids)

        for info in imgs:
            file_name = info.get("file_name")
            h, w = int(info["height"]), int(info["width"])
            img_path = os.path.join(IMAGES_DIR, file_name)
            if not os.path.exists(img_path):
                print(f"  [WARN] 画像見つからず: {img_path}")
                continue

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"  [WARN] 画像読込失敗: {img_path}")
                continue

            if (img.shape[0], img.shape[1]) != (h, w):
                print(f"  [WARN] サイズ不一致: JSON=({w}x{h}) / IMG=({img.shape[1]}x{img.shape[0]}) -> ズレの原因候補")

            # 面積トップK注釈を可視化
            ann_ids = coco.getAnnIds(imgIds=[info["id"]])
            anns = coco.loadAnns(ann_ids)

            # area 大きい順
            anns_sorted = sorted(anns, key=lambda a: a.get("area", 0), reverse=True)
            dbg = img.copy()
            H, W = img.shape[:2]

            for rank, ann in enumerate(anns_sorted[:TOPK], 1):
                m = coco.annToMask(ann).astype(np.uint8)  # 0/1
                frac = float(ann.get("area", 0)) / float(H*W + 1e-6)
                print(f"    ann_id={ann['id']:>6} cat={ann['category_id']:>3} area_ratio={frac:.3f}")

                # 輪郭も描く
                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(dbg, contours, -1, (0, 0, 255), 2)  # 輪郭=赤
                dbg = overlay_mask(dbg, m, alpha=0.35)               # 塗り=緑で半透明

            out_name = Path(file_name).stem + "_topK.png"
            cv2.imwrite(os.path.join(DBG_DIR, out_name), dbg)

if __name__ == "__main__":
    main()
