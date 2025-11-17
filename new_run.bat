:: Orden de ejecuciones

@echo on
call venv\Scripts\activate

cd segformer
call python segformer.py --mode "train" --min-pol 2 --size 256 --name "sf-400-256" --prepath "../data/raw/data_set_all_" --epochs 200
call python segformer.py --mode "val" --name "sf-400-256" --prepath "../data/raw/data_set_all_" --epochs 400
call python segformer.py --mode "train" --min-pol 1 --size 512 --name "sf-400-512" --prepath "../data/raw/data_set_all_" --epochs 200
call python segformer.py --mode "val" --name "sf-400-512" --prepath "../data/raw/data_set_all_" --epochs 400
call python segformer.py --mode "train" --min-pol 2 --size 256 --name "sf-400-256-overlap" --prepath "../data/raw/data_set_all_overlap_" --epochs 200
call python segformer.py --mode "val" --name "sf-400-256-overlap" --prepath "../data/raw/data_set_all_overlap_" --epochs 400
call python segformer.py --mode "train" --min-pol 1 --size 512 --name "sf-400-512-overlap" --prepath "../data/raw/data_set_all_overlap_" --epochs 200
call python segformer.py --mode "val" --name "sf-400-512-overlap" --prepath "../data/raw/data_set_all_overlap_" --epochs 400


cd ..
cd Yolo

call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 256 --epochs 200
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 256 --version 1
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 512 --epochs 200
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 512 --version 2
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 256 --epochs 200
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 256 --version 3
call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_overlap_" --size 512 --epochs 200
call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_overlap_" --size 512 --version 4


pause
