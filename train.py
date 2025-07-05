from ultralytics import YOLO


model = YOLO('yolov8n-seg.pt')

model.train(data='data.yaml',  
            epochs=50,           
            batch=8,             
            imgsz=640,           
            device='cpu')        
