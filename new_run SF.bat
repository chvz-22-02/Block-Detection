:: Orden de ejecuciones

@echo on
call ..\pi-env\Scripts\activate

@REM cd segformer

@REM call python segformer.py --mode "train" --min-pol 1 --size 256 --name "sf-2000-256-2" --prepath "../data/raw/data_set_all_" --epochs 25 --steps 2000
@REM call python segformer.py --mode "train" --min-pol 1 --size 256 --name "sf-2000-256-overlap-2" --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 2000
@REM call python segformer.py --mode "train" --min-pol 1 --size 512 --name "sf-2000-512-2" --prepath "../data/raw/data_set_all_" --epochs 25 --steps 2000
@REM call python segformer.py --mode "train" --min-pol 1 --size 512 --name "sf-2000-512-overlap-2" --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 2000
@REM call python segformer.py --mode "train" --min-pol 1 --size 1024 --name "sf-2000-1024-2" --prepath "../data/raw/data_set_all_" --epochs 25 --steps 2000
@REM call python segformer.py --mode "train" --min-pol 1 --size 1024 --name "sf-2000-1024-overlap-2" --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 2000

@REM call python segformer.py --mode "val" --name "sf-2000-256-2" --size 256 --prepath "../data/raw/data_set_all_" --epochs 5 --steps 400
@REM call python segformer.py --mode "val" --name "sf-2000-256-overlap-2" --size 256 --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 400
@REM call python segformer.py --mode "val" --name "sf-2000-512-2" --size 512 --prepath "../data/raw/data_set_all_"
@REM call python segformer.py --mode "val" --name "sf-2000-512-overlap-2" --size 512 --prepath "../data/raw/data_set_all_overlap_"
@REM call python segformer.py --mode "val" --name "sf-2000-1024-2" --size 1024 --prepath "../data/raw/data_set_all_"
@REM call python segformer.py --mode "val" --name "sf-2000-1024-overlap-2" --size 1024 --prepath "../data/raw/data_set_all_overlap_"

@REM cd ..
cd Yolo

@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --epochs 2000 --gen-data "si"
@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 1024 --epochs 2000 --gen-data "si"
@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 512 --epochs 2000 --gen-data "si"
@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 512 --epochs 2000 --gen-data "si"
@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 256 --epochs 2000 --gen-data "si"
@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 256 --epochs 2000 --gen-data "si"

@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --version 7
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 1024 --version 8
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 9
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 512 --version 10
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 256 --version 11
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 256 --version 12

@REM :: Yolo de Gustavo

@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 1024 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 512 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 512 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 256 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 256 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"

call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --version "" --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 1024 --version 2 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 3 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 512 --version 4 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 256 --version 5 --type-yolo "yolo11s-obb.pt"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 256 --version 6 --type-yolo "yolo11s-obb.pt"

pause
