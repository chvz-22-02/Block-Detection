:: Orden de ejecuciones

@echo on
call venv\Scripts\activate

cd ..
cd Yolo

call python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 256 --epochs 200 --nombre-yaml "dataAnto.yaml" --min-pol 1

call python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 256 --version 1

pause
