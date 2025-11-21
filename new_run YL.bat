:: Orden de ejecuciones

@echo on
call ..\pi-env\Scripts\activate

@REM cd segformer
@REM @REM call python segformer.py --mode "train" --min-pol 2 --size 256 --name "sf-2000-256" --prepath "../data/raw/data_set_all_" --epochs 25 --steps 2000
@REM call python segformer.py --mode "val" --name "sf-2000-256" --size 256 --prepath "../data/raw/data_set_all_" --epochs 5 --steps 400
@REM pause
@REM @REM call python segformer.py --mode "train" --min-pol 1 --size 256 --name "sf-2000-256-overlap" --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 2000
@REM call python segformer.py --mode "val" --name "sf-2000-256-overlap" --size 256 --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 400
@REM @REM @REM pause
@REM @REM call python segformer.py --mode "train" --min-pol 1 --size 512 --name "sf-2000-512" --prepath "../data/raw/data_set_all_" --epochs 25 --steps 2000
@REM call python segformer.py --mode "val" --name "sf-2000-512" --size 512 --prepath "../data/raw/data_set_all_"

@REM @REM call python segformer.py --mode "train" --min-pol 1 --size 512 --name "sf-2000-512-overlap" --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 2000
@REM call python segformer.py --mode "val" --name "sf-2000-512-overlap" --size 512 --prepath "../data/raw/data_set_all_overlap_"

@REM @REM call python segformer.py --mode "train" --min-pol 1 --size 1024 --name "sf-2000-1024" --prepath "../data/raw/data_set_all_" --epochs 25 --steps 2000
@REM call python segformer.py --mode "val" --name "sf-2000-1024" --size 1024 --prepath "../data/raw/data_set_all_"

@REM @REM call python segformer.py --mode "train" --min-pol 1 --size 1024 --name "sf-2000-1024-overlap" --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 2000
@REM call python segformer.py --mode "val" --name "sf-2000-1024-overlap" --size 1024 --prepath "../data/raw/data_set_all_overlap_"



@REM cd ..
cd Yolo

@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 256 --epochs 2000 --gen-data "si"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 256 --version ""

@REM @REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 512 --epochs 2000 --gen-data "si"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 512 --version 2

@REM @REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 1024 --epochs 2000 --gen-data "si"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 1024 --version 3

@REM @REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 256 --epochs 2000 --gen-data "si"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 256 --version 4

@REM @REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 512 --epochs 2000 --gen-data "si"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 5

@REM @REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --epochs 2000 --gen-data "si"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --version 6

:: Yolo de Gustavo

@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 256 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 256 --version "" --type-yolo "yolo11s-obb.pt"

@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 512 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 512 --version 2 --type-yolo "yolo11s-obb.pt"

@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 1024 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 1024 --version 3 --type-yolo "yolo11s-obb.pt"

@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 256 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 256 --version 4 --type-yolo "yolo11s-obb.pt"

@REM call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 512 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 5 --type-yolo "yolo11s-obb.pt"

call python YOLO_11.py --mode "pred" --version 6 --prepath "../data/raw/data_set_all_overlap_" --size 1024 --epochs 2000 --gen-data "si" --type-yolo "yolo11s-obb.pt"
@REM call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --version 6 --type-yolo "yolo11s-obb.pt"

pause
