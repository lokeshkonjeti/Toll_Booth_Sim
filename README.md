# Toll_Booth_Sim

Code to Detect Vehicle type, Detect license plate and Recognize plate numbers.

Pipeline: Detect Vehicles using pre-built (YOLOv8 COCO) -> Plate detection pre-built (Excisitng HuggingFace weights) -> use OCR (EasyOCR)

High-Level Flow:
1. Open a video/webcam
2. Detect **vehicles** with YOLOv8 (COCO-pretrained)
3. For each vehicle crop, detect a **license plate** with a pre-trained model from HuggingFace
4. OCR the plate crop with **EasyOCR**
5. Draw overlays, write an **annotated video**, and append structured records to a **JSONL** file

# Test on a video file
python main.py --video input.mp4 --out annotated.mp4 --jsonl events.jsonl

# Webcam
python main.py --video 0 --show
