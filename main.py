
"""
Code to Detect Vehicle type, Detect license plate and Recognize plate numbers.

Pipeline:
    Detect Vehicles using pre-built (YOLOv8 COCO) -> Plate detection pre-built (Koushim HF weights) -> use OCR (EasyOCR)

Usage:
    python demo_koushim_lp.py --video input.mp4 --out annotated.mp4 --jsonl events.jsonl

"""

import argparse
import os
import cv2
import json
from datetime import datetime, timezone
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import easyocr

COCO_VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def clip_box(box, w, h):
    x1, y1, x2, y2 = box
    return [max(0, int(x1)), max(0, int(y1)), min(w-1, int(x2)), min(h-1, int(y2))]

def expand_box(box, w, h, scale=0.12):
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * scale; y1 -= bh * scale
    x2 += bw * scale; y2 += bh * scale
    return clip_box([x1, y1, x2, y2], w, h)

def crop(img, box):
    x1, y1, x2, y2 = map(int, box)
    return img[y1:y2, x1:x2]

def draw_box(img, box, color=(0,255,0), label=None):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to video, 0 for webcam")
    ap.add_argument("--out", default="annotated.mp4", help="annotated video path")
    ap.add_argument("--jsonl", default="", help="Write JSON file for the detections")
    ap.add_argument("--device", default="", help="'' autodetect, 'cpu', or GPU id like '0'")
    ap.add_argument("--conf-veh", type=float, default=0.25, help="Vehicle confidence threshold")
    ap.add_argument("--conf-plate", type=float, default=0.30, help="Plate confidence threshold")
    ap.add_argument("--ocr-lang", default="en", help="EasyOCR language")
    ap.add_argument("--ocr-gpu", action="store_true", help="To use GPU for EasyOCR if available")
    ap.add_argument("--show", action="store_true", help="Preview window")
    ap.add_argument("--save-crops", action="store_true", help="Save plate numbers")
    ap.add_argument("--crops-dir", default="crops")
    return ap.parse_args()

def main():
    args = parse_args()

    plate_weights_path = hf_hub_download(
        repo_id="Koushim/yolov8-license-plate-detection",
        filename="best.pt",
        repo_type="model"
    )

    veh_model = YOLO("yolov8n.pt")
    plate_model = YOLO(plate_weights_path)

    langs = [s.strip() for s in args.ocr_lang.split(",") if s.strip()]
    reader = easyocr.Reader(langs, gpu=args.ocr_gpu)

    src = 0 if args.video.strip() == "0" else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    jsonl_fp = open(args.jsonl, "a", encoding="utf-8") if args.jsonl else None
    if args.save_crops:
        os.makedirs(args.crops_dir, exist_ok=True)

    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            veh_res = veh_model.predict(source=frame, conf=args.conf_veh, device=args.device, verbose=False)[0]
            veh_boxes = []
            for b, c, conf in zip(veh_res.boxes.xyxy.cpu().numpy(),
                                   veh_res.boxes.cls.cpu().numpy().astype(int),
                                   veh_res.boxes.conf.cpu().numpy()):
                if c in COCO_VEHICLE_IDS:
                    veh_boxes.append((b.tolist(), c, float(conf)))

            events = []
            for (vx1, vy1, vx2, vy2), vcls, vconf in veh_boxes:
                vbox = [vx1, vy1, vx2, vy2]
                vcrop = crop(frame, vbox)
                if vcrop.size == 0:
                    continue

                p_res = plate_model.predict(source=vcrop, conf=args.conf_plate, device=args.device, verbose=False)[0]

                best = None
                for pb, pc, pconf in zip(p_res.boxes.xyxy.cpu().numpy(),
                                         p_res.boxes.cls.cpu().numpy().astype(int),
                                         p_res.boxes.conf.cpu().numpy()):
                    if (best is None) or (pconf > best["conf"]):
                        best = {"box": pb.tolist(), "conf": float(pconf)}

                plate_text, ocr_conf, crop_path = "", None, None
                if best:
                    px1, py1, px2, py2 = best["box"]
                    px1 += vx1; py1 += vy1; px2 += vx2; py2 += vy2
                    pbox = clip_box([px1, py1, px2, py2], w, h)
                    pbox_exp = expand_box(pbox, w, h, scale=0.12)
                    pimg = crop(frame, pbox_exp)

                    if args.save_crops:
                        crop_path = os.path.join(args.crops_dir, f"f{frame_idx:06d}.jpg")
                        cv2.imwrite(crop_path, pimg)

                    ocr = reader.readtext(pimg, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ")
                    if ocr:
                        ocr.sort(key=lambda x: x[2], reverse=True)
                        plate_text = ocr[0][1].strip()
                        ocr_conf = float(ocr[0][2])

                    draw_box(frame, vbox, (0,255,0), f"{COCO_VEHICLE_IDS.get(vcls,'veh')} {vconf:.2f}")
                    draw_box(frame, pbox, (255,0,0), f"plate {best['conf']:.2f}")
                    if plate_text:
                        cv2.putText(frame, plate_text, (pbox[0], min(pbox[3]+22, h-8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2, cv2.LINE_AA)
                    events.append({
                        "timestamp_utc": now_utc_iso(),
                        "frame_index": frame_idx,
                        "vehicle_type": COCO_VEHICLE_IDS.get(vcls, "vehicle"),
                        "vehicle_conf": float(vconf),
                        "plate_conf": best["conf"],
                        "plate_text": plate_text,
                        "ocr_conf": ocr_conf,
                        "vehicle_bbox_xyxy": [int(vx1), int(vy1), int(vx2), int(vy2)],
                        "plate_bbox_xyxy": [int(pbox[0]), int(pbox[1]), int(pbox[2]), int(pbox[3])],
                        "crop_path": crop_path
                    })
                else:
                    draw_box(frame, vbox, (0,255,0), f"{COCO_VEHICLE_IDS.get(vcls,'veh')} {vconf:.2f}")

            writer.write(frame)
            if args.show:
                cv2.imshow("LP Demo", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if jsonl_fp and events:
                for ev in events:
                    jsonl_fp.write(json.dumps(ev) + "\n")
                jsonl_fp.flush()

    finally:
        cap.release()
        writer.release()
        if args.show:
            try: cv2.destroyAllWindows()
            except: pass
        if jsonl_fp:
            jsonl_fp.close()

if __name__ == "__main__":
    main()
