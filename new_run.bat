:: Orden de ejecuciones

@echo on
call ..\pi-env\Scripts\activate

cd segformer
call python segformer.py --mode "train" --min-pol 2 --size 256 --name "sf-800-256" --prepath "../data/raw/data_set_all_" --epochs 25 --steps 800
call python segformer.py --mode "val" --name "sf-800-256" --prepath "../data/raw/data_set_all_" --epochs 5 --steps 400

call python segformer.py --mode "train" --min-pol 1 --size 256 --name "sf-800-256" --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 800
call python segformer.py --mode "val" --name "sf-800-256" --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 400
@REM pause
call python segformer.py --mode "train" --min-pol 1 --size 512 --name "sf-800-512" --prepath "../data/raw/data_set_all_" --epochs 25 --steps 800
call python segformer.py --mode "val" --name "sf-800-512" --prepath "../data/raw/data_set_all_" --epochs 5 --steps 800

call python segformer.py --mode "train" --min-pol 1 --size 512 --name "sf-800-512-overlap" --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 800
call python segformer.py --mode "val" --name "sf-800-512-overlap" --prepath "../data/raw/data_set_all_overlap_" --epochs 5 --steps 800

call python segformer.py --mode "train" --min-pol 1 --size 1024 --name "sf-800-1024" --prepath "../data/raw/data_set_all_" --epochs 25 --steps 800
call python segformer.py --mode "val" --name "sf-800-1024" --prepath "../data/raw/data_set_all_" --epochs 5 --steps 800

call python segformer.py --mode "train" --min-pol 1 --size 1024 --name "sf-800-1024-overlap" --prepath "../data/raw/data_set_all_overlap_" --epochs 25 --steps 800
call python segformer.py --mode "val" --name "sf-800-1024-overlap" --prepath "../data/raw/data_set_all_overlap_" --epochs 5 --steps 800



cd ..
cd Yolo

call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 256 --epochs 400 --gen-data "si"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 256 --version ""

call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 512 --epochs 400 --gen-data "si"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 512 --version 2

call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 1024 --epochs 400 --gen-data "si"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 1024 --version 3

call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 256 --epochs 400 --gen-data "si"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 256 --version 4

call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 512 --epochs 400 --gen-data "si"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 5

call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --epochs 400 --gen-data "si"
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 1024 --version 6


pause
