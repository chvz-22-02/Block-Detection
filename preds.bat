
@echo on
call ..\pi-env\Scripts\activate

cd segformer

call python segformer.py --mode "pred" --name "sf-2000-256" --size 256 --prepath "../data/raw/data_set_all_"

call python segformer.py --mode "pred" --name "sf-2000-256-overlap" --size 256 --prepath "../data/raw/data_set_all_overlap_"

call python segformer.py --mode "pred" --name "sf-2000-512" --size 512 --prepath "../data/raw/data_set_all_"

call python segformer.py --mode "pred" --name "sf-2000-512-overlap" --size 512 --prepath "../data/raw/data_set_all_overlap_"

call python segformer.py --mode "pred" --name "sf-2000-1024" --size 1024 --prepath "../data/raw/data_set_all_"

call python segformer.py --mode "pred" --name "sf-2000-1024-overlap" --size 1024 --prepath "../data/raw/data_set_all_overlap_"

cd ..
cd Yolo


call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --version 7
call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_" --size 1024 --version 8
call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 9
call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_" --size 512 --version 10
call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_overlap_" --size 256 --version 11
call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_" --size 256 --version 12

call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --version "" --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_" --size 1024 --version 2 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 3 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_" --size 512 --version 4 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_overlap_" --size 256 --version 5 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_" --size 256 --version 6 --type-yolo "yolo11s-obb.pt"

@REM cd Roboflow

@REM call python YOLO_Roboflow.py --mode "pred" --size 640 --version ""

@REM call python YOLO_Roboflow.py --mode "pred" --size 640 --version 2

@REM call python YOLO_Roboflow.py --mode "pred" --size 640 --version 3

pause