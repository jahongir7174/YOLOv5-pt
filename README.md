[YOLOv5](https://github.com/ultralytics/yolov5) reimplementation using PyTorch

#### Train
* Create `train.txt` and `test.txt` files for your dataset
* Configure your dataset paths in `main.py`
* Run `python -m torch.distributed.launch --nproc_per_node $ main.py` for training, `$` is number of GPUs


#### Dataset structure
    ├── Dataset folder 
        ├── images
            ├── 1111.jpg
            ├── 2222.jpg
        ├── labels
            ├── 1111.txt
            ├── 2222.txt
        ├── train.txt
        ├── test.txt
        
#### Note
* txt file should be in YOLO format
* [this repo](https://github.com/jahongir7174/YOLO2VOC) can be used to VOC to YOLO format 

#### Reference
* https://github.com/ultralytics/yolov5