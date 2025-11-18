:: Orden de ejecuciones

@echo on
call ..\pi-env\Scripts\activate

cd Yolo

:: ENTRENAR
::python YOLO_11.py --mode "train" --prepath "../data/raw/data_set_all_" --size 256 --epochs 200 --nombre-yaml "dataAnto.yaml" --min-pol 1 --gen-data "no"

:: VALIDAR (AUTOMÁTICO - selecciona el último modelo)
python YOLO_11.py --mode "val" --prepath "../data/raw/data_set_all_" --size 256 --nombre-yaml "dataAnto.yaml" --version 2

pause
