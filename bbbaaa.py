# asdf-lkj2412
# demoの中


from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent))
sys.path.append(str(Path.cwd().parent / 'label_anything'))







from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from label_anything import LabelAnything
from label_anything.data import get_preprocessing, utils
from label_anything.data.transforms import PromptsProcessor

# ---- 0) 画像の指定（先頭=クエリ、次=サポート1枚） ----
img_dir = Path("demo/images")  # 変えてOK
img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) +
                   list(img_dir.glob("*.png")) + list(img_dir.glob("*.JPG")) + list(img_dir.glob("*.PNG")))
assert len(img_paths) >= 2, "クエリ1 + サポート1 以上の画像を置いてください"

open_rgb = lambda p: Image.open(p).convert("RGB")
query_p = img_paths[0]
support_ps = [img_paths[1]]           # 必要なら複数でもOK

query_img = open_rgb(query_p)
support_imgs = [open_rgb(p) for p in support_ps]

# ---- 1) 事前学習モデルをロード（学習はしない） ----
# 公式のプリトレ重みをそのまま使う
la = LabelAnything.from_pretrained("pasqualedem/label_anything_sam_1024_coco")  # :contentReference[oaicite:2]{index=2}
la.eval()

# ---- 2) 前処理 & プロンプト準備 ----
image_size = 1024
preprocess = get_preprocessing({"common": {"custom_preprocess": True, "image_size": image_size}})
query_t = preprocess(query_img)
support_t = [preprocess(im) for im in support_imgs]

# COCOのクラスID: 例として person=1 を推論対象にする（dog=18 などに変えてOK）
cat_ids = [-1, 1]  # -1 は背景
# サポート画像上の矩形（ピクセル）。[x1,y1,x2,y2] を自分の画像に合わせて変更
bbox_px = [50, 50, 300, 400]

# 変換（サポート画像ごとに用意）
proc = PromptsProcessor(long_side_length=image_size, masks_side_length=256, custom_preprocess=True)
raw_bboxes = []
for im in support_imgs:
    raw = {-1: [], 1: []}
    raw[1] = [proc.convert_bbox(bbox_px, *im.size, noise=False)]
    raw_bboxes.append(raw)

# numpy 化 → テンソル化
for d in raw_bboxes:
    for cid in cat_ids:
        d[cid] = np.array(d[cid])

bboxes, flag_bboxes = utils.annotations_to_tensor(proc, raw_bboxes,
                                                  [im.size for im in support_imgs],
                                                  utils.PromptType.BBOX)
flag_examples = utils.flags_merge(flag_bboxes=flag_bboxes)

# ---- 3) 入力パック → 推論 ----
batch = {
    utils.BatchKeys.IMAGES: torch.stack([query_t] + support_t).unsqueeze(0),  # (B=1, N=1+S, C,H,W)
    utils.BatchKeys.PROMPT_BBOXES: bboxes.unsqueeze(0),
    utils.BatchKeys.FLAG_BBOXES: flag_bboxes.unsqueeze(0),
    utils.BatchKeys.FLAG_EXAMPLES: flag_examples.unsqueeze(0),
    utils.BatchKeys.DIMS: torch.tensor([[im.size for im in [query_img] + support_imgs]]),
}
with torch.no_grad():
    out = la(batch)             # 学習しない＝forwardだけ
logits = out["logits"]          # (B, Classes, h, w)

# クエリ画像の解像度に合わせてアップサンプルしてから argmax
logits_up = F.interpolate(logits, size=query_img.size[::-1], mode="bilinear", align_corners=False)
pred = logits_up.argmax(1)[0].cpu().numpy()   # (H, W) クラスマップ

# ---- 4) 予測マスクを保存（ person=1 の部分だけ抽出） ----
mask = (pred == 1).astype(np.uint8) * 255
Image.fromarray(mask).save("pred_mask.png")
print("saved: pred_mask.png")
