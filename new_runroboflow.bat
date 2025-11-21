:: Orden de ejecuciones

@echo on
call ..\pi-env\Scripts\activate

cd Yolo\Roboflow

:: ENTRENAR
@REM call python YOLO_Roboflow.py --mode "train" --size 256 --epochs 2000 
@REM call python YOLO_Roboflow.py --mode "val" --size 256 --version ""
@REM call python YOLO_Roboflow.py --mode "pred" --size 256 --version ""

@REM call python YOLO_Roboflow.py --mode "train" --size 512 --epochs 2000
@REM call python YOLO_Roboflow.py --mode "val" --size 512 --version 2
@REM call python YOLO_Roboflow.py --mode "pred" --size 512 --version 2

@REM call python YOLO_Roboflow.py --mode "train" --size 640 --epochs 2000
@REM call python YOLO_Roboflow.py --mode "val" --size 640 --version 5
call python YOLO_Roboflow.py --mode "pred" --size 512 --version 3

pause
