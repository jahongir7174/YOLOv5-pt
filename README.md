[YOLOv5](https://github.com/ultralytics/yolov5) reimplementation using PyTorch

#### Train
* Create `train.txt` and `test.txt` files for your dataset
* Configure your dataset paths in `main.py`
* Run `python -m torch.distributed.launch --nproc_per_node $ main.py` for training, `$` is number of GPUs


#### Dataset structure
    ├── Dataset folder 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt
        ├── train2017.txt
        ├── test2017.txt
        
#### Note
* labels/*.txt files should be `cls, x1y1, x2y2, ... xnyn` normalized polygon format

#### Reference
* https://github.com/ultralytics/yolov5