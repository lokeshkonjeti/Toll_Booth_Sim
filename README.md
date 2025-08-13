# Toll_Booth_Sim

Code to Detect Vehicle type, Detect license plate and Recognize plate numbers.

Pipeline: Detect Vehicles using pre-built (YOLOv8 COCO) -> Plate detection pre-built (Excisitng HuggingFace weights) -> use OCR (EasyOCR)

**High-Level Flow:**
1. Open a video/webcam
2. Detect **vehicles** with YOLOv8 (COCO-pretrained)
3. For each vehicle crop, detect a **license plate** with a pre-trained model from HuggingFace
4. OCR the plate crop with **EasyOCR**
5. Draw overlays, write an **annotated video**, and append structured records to a **JSONL** file

**Helper functions:**
clip_box(box, w, h) → clamps a bbox to image bounds (no negative or out-of-frame coords).

expand_box(box, w, h, scale=0.12) → grows the bbox by a % on each side. This can be helpful for OCR (as it can give the reader a little area around the plate to validate better).

crop(img, box) → returns the image region for the bbox.

draw_box(img, box, color, label) → draws a rectangle + text label on the frame.

now_utc_iso() → ISO 8601 timestamp in UTC for logging.

**CLI arguments (parse_args):**
--video → path, or 0 for webcam.

--out → annotated video output path (annotated.mp4).

--jsonl → if set, append one JSON object per detection here.

--device → '' (auto), 'cpu', or GPU id like '0'.

--conf-veh, --conf-plate → confidence thresholds for the two detectors.

--ocr-lang → EasyOCR languages (e.g., "en" or "en,pt").

--ocr-gpu → use GPU for EasyOCR if available.

--show → live preview window (Esc to quit).

--save-crops, --crops-dir → save OCR plate crops for QA/datasets.

# Test on a video file
python main.py --video input.mp4 --out annotated.mp4 --jsonl events.jsonl

# Webcam
python main.py --video 0 --show

# Save the cropped plate images for manual validation
python main.py --video input.mp4 --save-crops --crops-dir crops
