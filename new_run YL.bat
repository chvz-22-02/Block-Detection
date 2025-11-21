:: Orden de ejecuciones

@echo on
call ..\pi-env\Scripts\activate

cd Yolo

call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --epochs 2000 --gen-data "si"
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 1024 --epochs 2000 --gen-data "si"
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 512 --epochs 2000 --gen-data "si"
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 512 --epochs 2000 --gen-data "si"
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 256 --epochs 2000 --gen-data "si"
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 256 --epochs 2000 --gen-data "si"

call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --version 7
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 1024 --version 8
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 9
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 512 --version 10
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 256 --version 11
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 256 --version 12

:: Yolo de Gustavo

call python YOLO_11.py --mode "pred" --version 6 --prepath "../data/raw/data_set_all_overlap_" --size 1024 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 1024 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 512 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 512 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 256 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 256 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"

call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --version "" --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 1024 --version 2 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 3 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 512 --version 4 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 256 --version 5 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 256 --version 6 --type-yolo "yolo11s-obb.pt"

pause
