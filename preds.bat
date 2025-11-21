
@echo on
call ..\pi-env\Scripts\activate

@REM cd segformer

@REM call python segformer.py --mode "pred" --name "sf-2000-256" --size 256 --prepath "../data/raw/data_set_all_"

@REM call python segformer.py --mode "pred" --name "sf-2000-256-overlap" --size 256 --prepath "../data/raw/data_set_all_overlap_"

@REM call python segformer.py --mode "pred" --name "sf-2000-512" --size 512 --prepath "../data/raw/data_set_all_"

@REM call python segformer.py --mode "pred" --name "sf-2000-512-overlap" --size 512 --prepath "../data/raw/data_set_all_overlap_"

@REM call python segformer.py --mode "pred" --name "sf-2000-1024" --size 1024 --prepath "../data/raw/data_set_all_"

@REM call python segformer.py --mode "pred" --name "sf-2000-1024-overlap" --size 1024 --prepath "../data/raw/data_set_all_overlap_"

@REM cd ..
cd Yolo


call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_" --size 256 --version ""

call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_" --size 512 --version 2

call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_" --size 1024 --version 3

call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_overlap_" --size 256 --version 4

call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 5

call python YOLO_11.py --mode "pred" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 6

@REM cd Roboflow

@REM call python YOLO_Roboflow.py --mode "pred" --size 640 --version ""

@REM call python YOLO_Roboflow.py --mode "pred" --size 640 --version 2

@REM call python YOLO_Roboflow.py --mode "pred" --size 640 --version 3

pause