
import argparse
import os
from typing import Optional, List
import sys
from ultralytics import YOLO
import pandas as pd
import numpy as np
import shutil

# carga de utils
sys.path.append('../../utils')
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
    parser.add_argument("--nombre-yaml", type=str, default="data.yaml",
                        help="Ruta al data.yaml")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Umbral de confianza.")
    parser.add_argument("--max-det", type=int, default=300, help="Máximo de detecciones por imagen.")
    parser.add_argument("--classes", type=int, nargs="*", default=None,
                        help="Lista opcional de índices de clase a filtrar, e.g. --classes 0 2 5")
    parser.add_argument("--iou", type=float, default=0.5, help="Umbral IoU para emparejamiento.")
    parser.add_argument("--version", type=str, default="12", help="Versión (string suelto).")
    parser.add_argument("--gen-data", type=str, default="si", help="Si es que se tiene que generar el dataset o buscarlo directamente al yaml (bool).")

    return parser.parse_args()


def main():
    args = parse_args()
    epochs = args.epochs
    min_pol = args.min_pol
    size = args.size
    n_modelo = args.version
    prepath = args.prepath
    gen_data = args.gen_data
    nombre_yaml = args.nombre_yaml

    model_path = f"runs/detect/train{n_modelo}/weights/best.pt"
    data_yaml  = f'{nombre_yaml}'
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
    path = f'{size}/'
    print(F'Lote de trabajo: {path}')

    if mode=='train':
        # Definición del dataset y separación en train, test y val

        # Carga del modelo base
        model = YOLO("../yolo11s.pt")

        # Fine tunning
        data_path = f"{size}.yaml"
        results= model.train (data=data_path,
        epochs=epochs,
        imgsz=size, 
        augment=True,
        patience=30)
        print('''
              -----------------------------------------------------------------------------------------------------
              -----------------------------------------------------------------------------------------------------
              ----------------------------------TERMINO EL TRAIN --------------------------------------------------
              -----------------------------------------------------------------------------------------------------
              -----------------------------------------------------------------------------------------------------''')


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
        
        data_path = f"{size}.yaml"
        with open(data_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        val_images_dir = data.get("test")
        model = YOLO(model_path)
        predicciones = model.predict(
            source=val_images_dir, imgsz=size, conf=conf_thres,
            max_det=max_det, classes=classes, stream=True,
            device=model.device, verbose=False, save=False
        )

        names_map = model.names
        if isinstance(names_map, dict):
            names = [names_map[i] for i in sorted(names_map.keys())]
        else:
            names = list(names_map)
        nc = len(names)

        cm = ConfusionMatrix(names=names)  # mantenemos ConfusionMatrix para TP/FP/FN

        img_count = 0
        pred_box_count = 0
        gt_box_count = 0

        # Acumuladores SOLO para IoU de verdaderos positivos por clase (con conf >= conf_thres)
        sum_iou_tp_per_cls = np.zeros((nc,), dtype=np.float64)
        count_tp_per_cls   = np.zeros((nc,), dtype=np.int64)

        for res in predicciones:
            img_count += 1
            img_path = getattr(res, "path", None)
            oh, ow = getattr(res, "orig_shape", (None, None))

            # Detecciones del modelo
            det_xyxy, det_conf, det_cls = extract_pred_xyxy_conf_cls(res)
            pred_box_count += det_xyxy.shape[0]

            # Ground truth
            label_path = (
                img_path.replace("images", "labels")
                        .replace(".png", ".txt")
                        .replace(".jpg", ".txt")
                        .replace(".jpge", ".txt")
            )
            gt_xyxy, gt_cls = load_gt_xyxy_and_cls(label_path, img_w=ow, img_h=oh)
            gt_box_count += gt_xyxy.shape[0]

            # === Matriz de confusión (Ultralytics) ===
            # OJO: aquí ya pasamos conf=conf_thres e iou=iou, así que P/R se calculan con ese filtro
            update_cm(
                cm,
                det_xyxy=det_xyxy, det_conf=det_conf, det_cls=det_cls,
                gt_xyxy=gt_xyxy,   gt_cls=gt_cls,
                iou=iou, conf=conf_thres
            )

            # === IoU por clase (solo TP), usando el MISMO filtro de confianza ===
            # ### NUEVO: aplicamos filtro de confianza ANTES del matching
            if det_xyxy.size:
                keep = det_conf >= conf_thres     # <<— mismo umbral que usa cm.process_batch
                det_xyxy_f = det_xyxy[keep]
                det_cls_f  = det_cls[keep]
            else:
                det_xyxy_f = det_xyxy
                det_cls_f  = det_cls

            if det_xyxy_f.size and gt_xyxy.size:
                matches = match_dets_to_gt_greedy(det_xyxy_f, det_cls_f, gt_xyxy, gt_cls, iou_thres=iou)
            else:
                matches = []

            for di, gi, iou_val in matches:
                c = int(det_cls_f[di])
                if 0 <= c < nc:
                    sum_iou_tp_per_cls[c] += float(iou_val)
                    count_tp_per_cls[c]   += 1

        # ---------- Métricas desde la matriz de confusión ----------
        M = cm.matrix  # Ultralytics: nc x nc (no incluye "fondo" explícito)
        eps = 1e-12

        per_class_metrics = []
        for i in range(nc):
            TP_i = M[i, i]
            FP_i = M[:, i].sum() - TP_i   # todo lo que se predijo como i y no era i
            FN_i = M[i, :].sum() - TP_i   # todo lo que era i y no se predijo i

            prec_i = TP_i / (TP_i + FP_i + eps)
            rec_i  = TP_i / (TP_i + FN_i + eps)

            # IoU por clase (promedio de IoUs de TPs con conf>=conf_thres)
            iou_i = float(sum_iou_tp_per_cls[i] / max(count_tp_per_cls[i], 1)) if count_tp_per_cls[i] > 0 else 0.0

            per_class_metrics.append({
                "class_idx": i,
                "class_name": names[i] if i < len(names) else str(i),
                "TP": float(TP_i), "FP": float(FP_i), "FN": float(FN_i),
                "precision": float(prec_i), "recall": float(rec_i),
                "iou_TP_mean": iou_i,
                "TP_count_for_IoU": int(count_tp_per_cls[i]),
            })

        # ---------- Promedios de IoU ----------
        # MICRO: ponderado por #TP (con conf>=conf_thres)
        total_TP_for_IoU = int(np.sum(count_tp_per_cls))
        mean_iou_micro = float(np.sum(sum_iou_tp_per_cls) / total_TP_for_IoU) if total_TP_for_IoU > 0 else 0.0

        # MACRO: promedio simple sobre clases con al menos 1 TP (con conf>=conf_thres)
        valid_iou_vals = [m["iou_TP_mean"] for m in per_class_metrics if m["TP_count_for_IoU"] > 0]
        mean_iou_macro = float(np.mean(valid_iou_vals)) if len(valid_iou_vals) > 0 else 0.0

        # ---------- Promedios de Prec/Rec (micro sobre clases de objeto) ----------
        TP_sum = float(np.trace(M))
        FP_sum = float(M.sum(axis=0).sum() - np.trace(M))  # suma de FP sobre columnas
        FN_sum = float(M.sum(axis=1).sum() - np.trace(M))  # suma de FN sobre filas

        precision_micro = TP_sum / (TP_sum + FP_sum + eps)
        recall_micro    = TP_sum / (TP_sum + FN_sum + eps)

        # ---------- Indicadores tipo “fondo” (opcional, no estándar en detección) ----------
        # En detección a nivel instancia no hay TN bien definido. Si igual querés cuantificar
        # lo relacionado al fondo, podés reportar:
        #   FP_obj = predijo objeto donde no había GT (columna de la clase, excepto diagonal)
        #   FN_obj = había GT objeto y no lo detectó (fila de la clase, excepto diagonal)
        # Para 1 clase:
        if nc == 1:
            FP_obj = float(M[:, 0].sum() - M[0, 0])
            FN_obj = float(M[0, :].sum() - M[0, 0])
            fondo_lines = [
                "== Indicadores relacionados a fondo (no estándar) ==",
                f"FP_obj (predijo objeto sin GT) = {int(FP_obj)}",
                f"FN_obj (no detectó objeto GT)  = {int(FN_obj)}",
            ]
        else:
            fondo_lines = []

        # ---------- Salida ----------
        texto_lineas = []
        texto_lineas.append(f"Imágenes evaluadas: {img_count}")
        texto_lineas.append(f"Detecciones totales: {pred_box_count}, GT totales: {gt_box_count}")
        texto_lineas.append("")
        texto_lineas.append("== Métricas por clase (solo objeto(s)) ==")
        for m in per_class_metrics:
            texto_lineas.append(
                f"[{m['class_idx']} - {m['class_name']}] "
                f"TP={m['TP']:.0f}, FP={m['FP']:.0f}, FN={m['FN']:.0f} | "
                f"Prec={m['precision']:.4f}, Rec={m['recall']:.4f}, "
                f"IoU_TP_mean={m['iou_TP_mean']:.4f} (TPs_IoU={m['TP_count_for_IoU']})"
            )
        if fondo_lines:
            texto_lineas.append("")
            texto_lineas += fondo_lines

        texto_lineas.append("")
        texto_lineas.append("== Promedios ==")
        texto_lineas.append(f"Precision_obj(micro)={precision_micro:.4f}, Recall_obj(micro)={recall_micro:.4f}")
        texto_lineas.append(
            f"IoU_micro(solo TP con conf≥{conf_thres}, ponderado por TP)={mean_iou_micro:.4f}, "
            f"IoU_macro(solo TP con conf≥{conf_thres}, clases con TP>0)={mean_iou_macro:.4f}"
        )

        ruta_archivo = f'../data/processed_yolo/stats/{path.split("/")[-2]}-Roboflow-{n_modelo}.txt'
        os.makedirs(os.path.dirname(ruta_archivo), exist_ok=True)
        with open(ruta_archivo, "w", encoding="utf-8") as f:
            f.write("\n".join(texto_lineas))

        print("\n".join(texto_lineas))
        print('''
              -----------------------------------------------------------------------------------------------------
              -----------------------------------------------------------------------------------------------------
              ----------------------------------TERMINO EL VAL-- --------------------------------------------------
              -----------------------------------------------------------------------------------------------------
              -----------------------------------------------------------------------------------------------------''')

        
    elif mode=='pred':
        # Carga del modelo entrenado
        custom_model = YOLO(f"runs/detect/train{version}/weights/best.pt")
        results = custom_model.predict(
            source="../../data/external",
            save=True,
            project="../data/processed_yolo",
            name=f"pred-m-Roboflow-{version}",
            exist_ok=True
        )
        print('''
              -----------------------------------------------------------------------------------------------------
              -----------------------------------------------------------------------------------------------------
              ----------------------------------TERMINO EL PRED ---------------------------------------------------
              -----------------------------------------------------------------------------------------------------
              -----------------------------------------------------------------------------------------------------''')
if __name__ == "__main__":
    main()