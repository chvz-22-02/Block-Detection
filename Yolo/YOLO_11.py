
import argparse
import os
from typing import Optional, List
import sys
from ultralytics import YOLO
import pandas as pd
import numpy as np
import shutil

# carga de utils
sys.path.append('../utils')
from utils import generar_lineas_por_manzana, procesar_jsons_en_carpeta, tabulate_jsons_from_folder, copiar_archivos_seleccionados


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Función de utilización de Yolo para entrenamiento, predicción y validación."
    )
    # Variables base
    parser.add_argument("--mode", type=str, default="pred", choices=["train", "val", "pred"],
                        help="Modo de ejecución (train/val/test/predict).")
    parser.add_argument("--epochs", type=int, default=200, help="Número de épocas de entrenamiento máximas (int).")
    parser.add_argument("--min-pol", type=int, default=2, help="Mínimo de polígonos (int).")
    parser.add_argument("--size", type=int, default=256, help="Tamaño base (int).")
    parser.add_argument("--prepath", type=str, default="../data/raw/data_set_", help="Ruta base previa (str).")
    parser.add_argument("--data-yaml", type=str, default="custom_object_detector_yolo11_v2-1/data.yaml",
                        help="Ruta al data.yaml")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Umbral de confianza.")
    parser.add_argument("--max-det", type=int, default=300, help="Máximo de detecciones por imagen.")
    parser.add_argument("--classes", type=int, nargs="*", default=None,
                        help="Lista opcional de índices de clase a filtrar, e.g. --classes 0 2 5")
    parser.add_argument("--iou", type=float, default=0.5, help="Umbral IoU para emparejamiento.")
    parser.add_argument("--version", type=str, default="12", help="Versión (string suelto).")

    return parser.parse_args()


def main():
    args = parse_args()
    epochs = args.epochs
    min_pol = args.min_pol
    size = args.size
    n_modelo = args.version
    prepath = args.prepath

    model_path = f"runs/detect/train{n_modelo}/weights/best.pt"
    data_yaml  = args.data_yaml
    conf_thres = args.conf_thres
    max_det    = args.max_det
    classes    = args.classes
    iou        = args.iou

    version = args.version
    mode = args.mode
    #################################################################################
    #################################################################################

    # carga de librerías



    # Ruta a la carpeta del dataset
    path = f'{prepath}{size}/'
    print(F'Ruta de trabajo: {path}')

    if mode=='train':
        # Definición del dataset y separación en train, test y val
        jsons = tabulate_jsons_from_folder(f'{path}metadata_y_{size}/')
        l1 = jsons[jsons['num_polygons_in_window']>1]
        l0 = jsons[jsons['num_polygons_in_window']==0].sample(n=int(np.round(len(l1)/5)), random_state=42)
        l = pd.concat([l1,l0])
        df_dict = l['bounds'].apply(pd.Series)
        l = pd.concat([l.drop(columns=['bounds']), df_dict], axis=1)

        l_val = l[((l['minx']>619641.7) & (l['miny']>9259735.0) & # V1 P1
                (l['minx']<621212.1) & (l['miny']<9261887.8)   # V2 P1
                ) |
                ((l['minx']>627583.9) & (l['miny']>9253353.3) & # V1 P2
                (l['minx']<629255) & (l['miny']<9255705)   # V2 P2
                ) |
                ((l['minx']>628000.9) & (l['miny']>9249430.7) & # V1 P3
                (l['minx']<620328.1) & (l['miny']<9239286.6)   # V2 P3
                ) |
                ((l['minx']>625001.3) & (l['miny']>9236962.5) & # V1 P4
                (l['minx']<625015.2) & (l['miny']<9236364.8)   # V2 P4
                ) |
                ((l['minx']>619685.9) & (l['miny']>9238780.6) & # V1 P5
                (l['minx']<620344.8) & (l['miny']<9239286.6)   # V2 P5
                )]['id']

        l_test = l[((l['minx']>618276.5) & (l['miny']>9256975.6) & # V1 P1
                    (l['minx']<623241.5) & (l['miny']<9252569.3)   # V2 P1
                    ) |
                    ((l['minx']>626004.8) & (l['miny']>9250947.2) & # V1 P2
                    (l['minx']<627251.6) & (l['miny']<9251964.7)   # V2 P2
                    ) |
                    ((l['minx']>633129.9) & (l['miny']>9251417.0) & # V1 P3
                    (l['minx']<634072.3) & (l['miny']<9251993.9)   # V2 P3
                    ) |
                    ((l['minx']>626975.0) & (l['miny']>9246374.1) & # V1 P4
                    (l['minx']<628832.1) & (l['miny']<9248253.4)   # V2 P4
                    ) |
                    ((l['minx']>617904.0) & (l['miny']>9243808.2) & # V1 P5
                    (l['minx']<619191.1) & (l['miny']<9245320.5)   # V2 P5
                    ) |
                    ((l['minx']>619124.4) & (l['miny']>9239440.9) & # V1 P6
                    (l['minx']<619783.2) & (l['miny']<9240155.3)   # V2 P6
                    ) |
                    ((l['minx']>630294.3) & (l['miny']>9240511.1) & # V1 P7
                    (l['minx']<632724) & (l['miny']<9242035)   # V2 P7
                    ) |
                    ((l['minx']>632418.2) & (l['miny']>9237497.6) & # V1 P8
                    (l['minx']<633908.3) & (l['miny']<9238604.1)   # V2 P8
                    )]['id']

        l_train = l['id'].drop(l_test.index).drop(l_val.index)

        # Motado de directorio (raw a interim)
        procesar_jsons_en_carpeta(f'{path}metadata_y_{size}/','../data/interim_yolo/dataset_labels/')
        copiar_archivos_seleccionados(f'{path}dataset_x_{size}/', '../data/interim_yolo/train/images/', [str(x)+'.png' for x in l_train])
        copiar_archivos_seleccionados(f'{path}dataset_x_{size}/', '../data/interim_yolo/valid/images/', [str(x)+'.png' for x in l_test])
        copiar_archivos_seleccionados(f'{path}dataset_x_{size}/', '../data/interim_yolo/test/images/', [str(x)+'.png' for x in l_val])
        copiar_archivos_seleccionados('../data/interim_yolo/dataset_labels/', '../data/interim_yolo/train/labels/', [str(x)+'.txt' for x in l_train])
        copiar_archivos_seleccionados('../data/interim_yolo/dataset_labels/', '../data/interim_yolo/valid/labels/', [str(x)+'.txt' for x in l_test])
        copiar_archivos_seleccionados('../data/interim_yolo/dataset_labels/', '../data/interim_yolo/test/labels/', [str(x)+'.txt' for x in l_val])

        # Carga del modelo base
        model = YOLO("yolo11s.pt")

        # Fine tunning
        data_path = "custom_object_detector_yolo11_v2-1/data.yaml"
        results= model.train (data=data_path,
        epochs=epochs,
        imgsz=size, 
        augment=False,
        patience=30,    
        early_stop=True)


    elif mode=='val':
        import yaml
        from typing import Any, Tuple, List
        import torch
        from ultralytics.utils.metrics import ConfusionMatrix

        def box_iou_np(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
            x11, y11, x12, y12 = boxes1[:, 0:1], boxes1[:, 1:2], boxes1[:, 2:3], boxes1[:, 3:4]
            x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]
            inter_x1 = np.maximum(x11, x21)
            inter_y1 = np.maximum(y11, y21)
            inter_x2 = np.minimum(x12, x22)
            inter_y2 = np.minimum(y12, y22)
            inter_w = np.clip(inter_x2 - inter_x1, 0, None)
            inter_h = np.clip(inter_y2 - inter_y1, 0, None)
            inter_area = inter_w * inter_h
            area1 = (x12 - x11) * (y12 - y11)
            area2 = (x22 - x21) * (y22 - y21) 
            union = area1 + area2 - inter_area 
            iou = np.where(union > 0, inter_area / union, 0.0).astype(np.float32)
            return iou

        def match_dets_to_gt_greedy(
            det_xyxy: np.ndarray,
            det_cls: np.ndarray,
            gt_xyxy: np.ndarray,
            gt_cls: np.ndarray,
            iou_thres: float
        ) -> List[Tuple[int, int, float]]:
            matches: List[Tuple[int, int, float]] = []
            ious = box_iou_np(det_xyxy, gt_xyxy)  
            valid = (det_cls[:, None] == gt_cls[None, :])
            ious_masked = np.where(valid, ious, -1.0) 
            cand_det, cand_gt = np.where(ious_masked >= iou_thres)

            cand_iou = ious_masked[cand_det, cand_gt]
            order = np.argsort(-cand_iou)

            assigned_dets = set()
            assigned_gts = set()

            for k in order:
                di = int(cand_det[k]); gi = int(cand_gt[k]); iou = float(cand_iou[k])
                if di in assigned_dets or gi in assigned_gts:
                    continue
                # Asignar este par
                assigned_dets.add(di); assigned_gts.add(gi)
                matches.append((di, gi, iou))

            return matches

        def polygon_norm_to_bbox_xyxy(coords_norm: np.ndarray, img_w: int, img_h: int):
            xs = coords_norm[0::2] * img_w
            ys = coords_norm[1::2] * img_h
            x1, y1 = xs.min(), ys.min()
            x2, y2 = xs.max(), ys.max()
            return np.array([[max(0, x1), max(0, y1), min(img_w - 1, x2), min(img_h - 1, y2)]], dtype=np.float32)

        def load_gt_xyxy_and_cls(label_path: str, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
            bboxes, classes = [], []
            with open(label_path, "r", encoding="utf-8-sig") as f:
                for raw in f:
                    line = raw.split("#", 1)[0].strip()
                    toks = line.split()
                    cls = int(float(toks[0]))
                    # Segmentación (cls x1 y1 x2 y2 ...)
                    coords = np.array([float(t) for t in toks[1:]], dtype=np.float32)
                    if coords.size >= 8 and coords.size % 2 == 0:
                        xyxy = polygon_norm_to_bbox_xyxy(coords, img_w, img_h)
                        if xyxy.size:
                            bboxes.append(xyxy[0]); classes.append(cls)
            if not bboxes:
                return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
            gt_bboxes = np.stack(bboxes, axis=0).astype(np.float32)
            gt_cls = np.array(classes, dtype=np.int64).reshape(-1)
            return gt_bboxes, gt_cls

        def extract_pred_xyxy_conf_cls(result_obj: Any):
            boxes = getattr(result_obj, "boxes", None)
            xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
            conf = boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1)
            cls  = boxes.cls.detach().cpu().numpy().astype(np.int64).reshape(-1)
            return xyxy, conf, cls

        def update_cm(cm: ConfusionMatrix,
                    det_xyxy: np.ndarray, det_conf: np.ndarray, det_cls: np.ndarray,
                    gt_xyxy: np.ndarray, gt_cls: np.ndarray,
                    iou: float, conf: float) -> None:
            t_det_xyxy = torch.from_numpy(det_xyxy.astype(np.float32)) if det_xyxy.size else torch.zeros((0,4), dtype=torch.float32)
            t_det_conf = torch.from_numpy(det_conf.astype(np.float32)) if det_conf.size else torch.zeros((0,),  dtype=torch.float32)
            t_det_cls  = torch.from_numpy(det_cls.astype(np.int64))    if det_cls.size  else torch.zeros((0,),  dtype=torch.int64)

            t_gtb = torch.from_numpy(gt_xyxy.astype(np.float32)) if gt_xyxy.size else torch.zeros((0,4), dtype=torch.float32)
            t_gtc = torch.from_numpy(gt_cls.astype(np.int64))    if gt_cls.size  else torch.zeros((0,),  dtype=torch.int64)

            gt_dict      = {"bboxes": t_gtb, "cls": t_gtc}
            det_dict_conf = {"bboxes": t_det_xyxy, "conf": t_det_conf, "cls": t_det_cls}
            cm.process_batch(det_dict_conf, gt_dict, conf=conf, iou_thres=iou)

        with open(data_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        val_images_dir = data.get("test")
        model = YOLO(model_path)
        predicciones = model.predict(source=val_images_dir, imgsz=size, conf=conf_thres,
                                    max_det=max_det, classes=classes, stream=True,
                                    device=model.device, verbose=False, save=False)

        names_map = model.names
        nc = len(names_map)
        labels_with_bg = [names_map[i] for i in range(nc)] + ["(bg)"]

        cm = ConfusionMatrix(names=names_map)

        img_count = 0
        pred_box_count = 0
        gt_box_count = 0

        sum_iou_tp = 0.0
        count_tp   = 0

        sum_iou_tp_per_cls = np.zeros((nc,), dtype=np.float64)
        count_tp_per_cls   = np.zeros((nc,), dtype=np.int64)

        for res in predicciones:
            img_count += 1
            img_path = getattr(res, "path", None)
            oh, ow = getattr(res, "orig_shape", (None, None))
            det_xyxy, det_conf, det_cls = extract_pred_xyxy_conf_cls(res)
            pred_box_count += det_xyxy.shape[0]
            label_path = img_path.replace("images", "labels").replace(".png", ".txt")
            gt_xyxy, gt_cls = load_gt_xyxy_and_cls(label_path, img_w=ow, img_h=oh)
            gt_box_count += gt_xyxy.shape[0]
            update_cm(cm,
                    det_xyxy=det_xyxy, det_conf=det_conf, det_cls=det_cls,
                    gt_xyxy=gt_xyxy,   gt_cls=gt_cls,
                    iou=iou, conf=conf_thres)
            if det_xyxy.size and gt_xyxy.size:
                matches = match_dets_to_gt_greedy(det_xyxy, det_cls, gt_xyxy, gt_cls, iou_thres=iou)
                for di, gi, miou in matches:
                    sum_iou_tp += miou
                    count_tp   += 1
                    c = int(det_cls[di])
                    sum_iou_tp_per_cls[c] += miou
                    count_tp_per_cls[c]   += 1

        mean_iou_global = (sum_iou_tp / count_tp) if count_tp > 0 else float('nan')
        mean_iou_por_clase = np.array([
            (sum_iou_tp_per_cls[c] / count_tp_per_cls[c]) if count_tp_per_cls[c] > 0 else np.nan
            for c in range(nc)
        ], dtype=np.float64)
        TP = cm.matrix[0,0]
        FN = cm.matrix[0,1]
        FP = cm.matrix[1,0]
        texto = f'Recall: {TP / (TP + FN)}, Precision: {TP / (TP + FN)}, IoU: {mean_iou_global}'
        ruta_archivo = f'../data/processed_yolo/stats/{n_modelo}.txt'
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        with open(ruta_archivo, "w", encoding="utf-8") as f:
            f.write(texto)

    elif mode=='pred':
        # Carga del modelo entrenado
        custom_model = YOLO(f"runs/detect/train{version}/weights/best.pt")
        results = custom_model.predict(
            source="../data/external",
            save=True,
            project="../data/processed_yolo",
            name=f"pred-m-{version}",
            exist_ok=True
        )

if __name__ == "__main__":
    main()