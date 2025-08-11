# Toll_Booth_Sim

Code to Detect Vehicle type, Detect license plate and Recognize plate numbers.

Pipeline: Detect Vehicles using pre-built (YOLOv8 COCO) -> Plate detection pre-built (Koushim HF weights) -> use OCR (EasyOCR)

# Test on a video file
python main.py --video input.mp4 --out annotated.mp4 --jsonl events.jsonl

# Webcam
python main.py --video 0 --show
