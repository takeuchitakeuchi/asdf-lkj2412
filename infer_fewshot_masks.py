
# infer_fewshot_masks.py
from pathlib import Path
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from label_anything import LabelAnything
from label_anything.data import get_preprocessing, utils
from label_anything.data.transforms import PromptsProcessor

def open_rgb(p): return Image.open(p).convert("RGB")
def open_mask_bin(p):
    # 0/255 等の2値やグレースケールを 0/1 に
    m = Image.open(p).convert("L")
    return (np.array(m) > 0).astype(np.uint8)  # (H,W) 0/1

def main(
    query_path="demo/images/query.jpg",
    supports_dir="demo/supports",
    prompts_json="demo/prompts_masks.json",
    model_name="pasqualedem/label_anything_sam_1024_coco",
    image_size=1024,
    out_path="pred_index.png",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) モデル（学習なし=推論モード）
    la = LabelAnything.from_pretrained(model_name).to(device).eval()

    # 2) 画像ロード & 前処理
    query_img = open_rgb(query_path)
    supp_paths = sorted([p for p in Path(supports_dir).glob("*.*") if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}])
    assert len(supp_paths) > 0, "サポート画像が見つかりません"

    preprocess = get_preprocessing({"common": {"custom_preprocess": True, "image_size": image_size}})
    query_t = preprocess(query_img)
    supp_imgs = [open_rgb(p) for p in supp_paths]
    supp_t = [preprocess(im) for im in supp_imgs]

    # 3) マスクプロンプト（JSON定義）を読み込み
    """
    prompts_masks.json の例：
    {
      "cat_ids": [-1, 1, 2, 3],        // 背景(-1)＋独自クラスID（手すり=1, 中桟=2, 巾木=3 など）
      "per_image": [
        {
          "index": 0,                  // supports_dir の 0番目画像に対応
          "masks": {
            "1": ["demo/masks/s0_handrail.png"],  // この画像における手すりのバイナリマスク(0/255や0/1)
            "2": [],                                // 中桟は無しでもOK（空リスト）
            "3": []                                 // 巾木は無しでもOK
          }
        }
        // 2枚目のサポートがあれば続けて { "index": 1, "masks": {...} }
      ]
    }
    """
    meta = json.loads(Path(prompts_json).read_text())
    cat_ids = meta["cat_ids"]            # ← 出力チャネルの並びになる（0→cat_ids[0], 1→cat_ids[1], ...）

    # 4) 生マスク → テンソル化（PromptType.MASK のみ）
    proc = PromptsProcessor(long_side_length=image_size, masks_side_length=256, custom_preprocess=True)

    raw_list = []
    for per in meta["per_image"]:
        im = supp_imgs[per["index"]]
        raw = {cid: [] for cid in cat_ids}
        # 各クラスのマスク画像パスを読んで 0/1 配列に
        for k, mask_paths in per.get("masks", {}).items():
            cid = int(k)
            for mp in mask_paths:
                raw[cid].append(open_mask_bin(mp))   # (H,W) 0/1

        # numpy 化（空でもOK。annotations_to_tensor 側で扱われる）
        for cid in cat_ids:
            raw[cid] = np.array(raw[cid], dtype=object)  # 異なる個数に対応
        raw_list.append(raw)

    masks_t, flag_masks = utils.annotations_to_tensor(
        proc,
        raw_list,
        [im.size for im in supp_imgs],        # 元画像の (W,H) を渡す
        utils.PromptType.MASK
    )
    flag_examples = utils.flags_merge(flag_masks=flag_masks)

    # 5) バッチを組んで forward（few-shot推論）
    batch = {
        utils.BatchKeys.IMAGES: torch.stack([query_t] + supp_t).unsqueeze(0).to(device),  # (B=1, N=1+S, C,H,W)
        utils.BatchKeys.PROMPT_MASKS: masks_t.unsqueeze(0).to(device),
        utils.BatchKeys.FLAG_MASKS: flag_masks.unsqueeze(0).to(device),
        utils.BatchKeys.FLAG_EXAMPLES: flag_examples.unsqueeze(0).to(device),
        utils.BatchKeys.DIMS: torch.tensor([[im.size for im in [query_img] + supp_imgs]], dtype=torch.int32).to(device),
    }

    with torch.no_grad():
        logits = la(batch)["logits"]  # (B, Classes, h, w) — Classes は len(cat_ids)

    # 6) 元解像度へUP → argmax（チャネルインデックス画像として保存）
    logits_up = F.interpolate(logits, size=query_img.size[::-1], mode="bilinear", align_corners=False)
    pred = logits_up.argmax(1)[0].cpu().numpy().astype(np.uint8)  # 値=0..C-1（cat_ids の並び）

    Image.fromarray(pred).save(out_path)
    print("saved:", out_path)
    print("channel index -> class id mapping:", list(enumerate(cat_ids)))

if __name__ == "__main__":
    # 必要なら argparse/click に差し替えてもOK。まずは固定値で動作確認。
    main()
