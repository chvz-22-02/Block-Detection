size = 256
nombre_modelo = 'segformer-200-512'
prepath = '../data/raw/data_set_'
checkpoint = '1080'
mode = 'train'
min_pol = 2

path = f'{prepath}{size}/'
print(F'Ruta de trabajo: {path}')

import os
import json
import pandas as pd
from datasets import DatasetDict, Dataset
from PIL import Image
import numpy as np
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, TrainingArguments, Trainer, EarlyStoppingCallback
import evaluate
evaluate.logging.set_verbosity_error()
import torch
from torch import nn

import sys
sys.path.append('../utils')

from utils import tabulate_jsons_from_folder, load_image_mask_pairs

metric = evaluate.load("mean_iou")
def build_compute_metrics(id2label, image_processor, ignore_index=0, compute_precision=True):
    """
    compute_metrics para HF Trainer que devuelve SIEMPRE:
      - mean_iou (mIoU global)
      - recall por clase (renombrado desde per_category_accuracy)
      - iou por clase
      - (opcional) precision por clase
      - macro_recall, macro_precision, f1_macro
    """
    num_labels = len(id2label)

    def compute_metrics(eval_pred):
        with torch.no_grad():
            logits, labels = eval_pred  # labels: numpy (B, H, W)
            logits_tensor = torch.from_numpy(logits)

            # Redimensionar logits a tamaño de labels y tomar argmax (clase por píxel)
            pred = nn.functional.interpolate(
                logits_tensor,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).argmax(dim=1)

            pred_np = pred.detach().cpu().numpy()   # (B, H, W)
            labels_np = labels                      # (B, H, W) numpy

            # mIoU semántico con ignore_index
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                metrics = metric._compute(
                    predictions=pred_np,
                    references=labels_np,
                    num_labels=num_labels,
                    ignore_index=ignore_index,
                    reduce_labels=getattr(image_processor, "do_reduce_labels", False),
                )

            # Extraer y renombrar resultados por clase
            per_cat_recall = metrics.pop("per_category_accuracy").tolist()  # == recall por clase
            per_cat_iou = metrics.pop("per_category_iou").tolist()

            # Publicar SIEMPRE recall e IoU por clase
            for i, v in enumerate(per_cat_recall):
                metrics[f"recall_{id2label[i]}"] = float(v)
            for i, v in enumerate(per_cat_iou):
                metrics[f"iou_{id2label[i]}"] = float(v)

            # (Opcional) Precision por clase y agregados macro
            if compute_precision:
                # Enmascarar píxeles inválidos
                mask_valid = (labels_np != ignore_index)
                preds_valid = pred_np[mask_valid]
                gts_valid = labels_np[mask_valid]

                per_cat_precision = []
                for c in range(num_labels):
                    tp = np.sum((preds_valid == c) & (gts_valid == c))
                    fp = np.sum((preds_valid == c) & (gts_valid != c))
                    prec = tp / (tp + fp + 1e-12)
                    per_cat_precision.append(prec)
                    metrics[f"precision_{id2label[c]}"] = float(prec)

                macro_recall = float(np.mean(per_cat_recall))
                macro_precision = float(np.mean(per_cat_precision))
                f1_macro = float(2 * macro_precision * macro_recall / (macro_precision + macro_recall + 1e-12))

                metrics["macro_recall"] = macro_recall
                metrics["macro_precision"] = macro_precision
                metrics["f1_macro"] = f1_macro

            # mean_iou ya viene en metrics['mean_iou'] (dejamos el nombre tal cual)
            return metrics
    return compute_metrics

if mode == 'train':
    jsons = tabulate_jsons_from_folder(f'{path}metadata_y_{size}/')
    l1 = jsons[jsons['num_polygons_in_window']>=min_pol]['id']
    l0 = jsons[jsons['num_polygons_in_window']==0]['id'].sample(n=int(np.round(len(l1)/5)), random_state=42)
    l = pd.concat([l1,l0])
    l_train_test = l.sample(frac=0.9, random_state=42)
    l_val = l.drop(l_train_test.index)
    l_train = l_train_test.sample(frac=0.8, random_state=42)
    l_test = l_train_test.drop(l_train.index)

    train_data = load_image_mask_pairs(f"{path}dataset_x_{size}/",f"{path}dataset_y_{size}/",l_train)
    val_data = load_image_mask_pairs(f"{path}dataset_x_{size}/",f"{path}dataset_y_{size}/",l_val)
    test_data = load_image_mask_pairs(f"{path}dataset_x_{size}/",f"{path}dataset_y_{size}/",l_test)

    dataset = DatasetDict({
        "train": Dataset.from_list(train_data),
        "test": Dataset.from_list(train_data),
        "validation": Dataset.from_list(val_data)
    })

    val_ds = dataset["validation"]
    test_ds = dataset['test']
    train_ds = dataset['train']

    image_processor = SegformerImageProcessor()

    # Sin Albumentations
    def train_transforms(example_batch):
        images = [np.array(image) for image in example_batch['pixel_values']]
        labels = [np.array(label) for label in example_batch['label']]
        inputs = image_processor(images, labels)
        return inputs

    def val_transforms(example_batch):
        images = [x for x in example_batch['pixel_values']]
        labels = [x for x in example_batch['label']]
        inputs = image_processor(images, labels)
        return inputs

    # Set transforms
    train_ds.set_transform(train_transforms)
    test_ds.set_transform(val_transforms)

    filename = "id2label.json"

    with open(filename, "r", encoding="utf-8") as f:
        id2label = json.load(f)

    #id2label = json.load(open(hf_hub_download(repo_id=sidewalk_dataset_identifier, filename=filename, repo_type="dataset"), "r"))
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}

    pretrained_model_name = "nvidia/mit-b0"
    model = SegformerForSemanticSegmentation.from_pretrained(
        pretrained_model_name,
        id2label=id2label,
        label2id=label2id
    )

    output_dir = "segformer-finetuned-200"
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=6e-5,
        num_train_epochs=200,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=20,
        eval_steps=20,
        logging_steps=1,
        eval_accumulation_steps=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,          
        load_best_model_at_end=True,
        push_to_hub=False,
        report_to=None
    )

    evaluate.logging.set_verbosity_error()

    # Carga el métrico de mIoU de Hugging Face
    metric = evaluate.load("mean_iou")

    compute_metrics = build_compute_metrics(
        id2label=id2label,
        image_processor=image_processor,
        ignore_index=0,             # ajusta según tu dataset
        compute_precision=True
    )

    # 2) Define TrainingArguments con evaluación y selección por métrica
    training_args = TrainingArguments(
        output_dir="out/segformer",
        evaluation_strategy="steps",   # o "epoch"
        eval_steps=500,                # ajusta al tamaño de tu dataset
        save_strategy="steps",         # guarda checkpoints sincronizados con eval
        save_steps=500,
        load_best_model_at_end=False,   # carga el mejor al final
        metric_for_best_model="mean_iou",  # puedes usar "mean_iou" si prefieres
        greater_is_better=True,        # True para f1_macro y mean_iou
        logging_steps=100,
        save_total_limit=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        seed=42,
    )

    # 3) Early stopping basado en la métrica de evaluación
    callbacks = [EarlyStoppingCallback(early_stopping_patience=30)]
    # Tip: si quieres early stopping más sensible, baja la paciencia (e.g., 10)

    # 4) Entrenador
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        tokenizer=image_processor,     # correcto para modelos vision en HF
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # 5) Entrenar y evaluar
    trainer.train()
    metrics = trainer.evaluate()
    print(metrics)  # contendrá mean_iou, f1_macro, macro_precision, macro_recall y por-clase

elif mode=='val':
    from transformers import AutoModelForSemanticSegmentation, Trainer, SegformerImageProcessor, TrainingArguments
    import json

    jsons = tabulate_jsons_from_folder(f'{path}metadata_y_{size}/')
    l1 = jsons[jsons['num_polygons_in_window']>=min_pol]['id']
    l0 = jsons[jsons['num_polygons_in_window']==0]['id'].sample(n=int(np.round(len(l1)/5)), random_state=42)
    l = pd.concat([l1,l0])
    l_train_test = l.sample(frac=0.9, random_state=42)
    l_val = l.drop(l_train_test.index)
    l_train = l_train_test.sample(frac=0.8, random_state=42)
    l_test = l_train_test.drop(l_train.index)

    val_data = load_image_mask_pairs(f"{path}dataset_x_{size}/",f"{path}dataset_y_{size}/",l_val)
    
    dataset = DatasetDict({
        "validation": Dataset.from_list(val_data)
    })

    val_data = load_image_mask_pairs(f"{path}dataset_x_{size}/",f"{path}dataset_y_{size}/",l_val)
    val_ds = dataset["validation"]

    # Ruta al checkpoint entrenado
    local_model_path = "segformer-finetuned/checkpoint-1080"

    # 1) Cargar el modelo desde el checkpoint
    model = AutoModelForSemanticSegmentation.from_pretrained(local_model_path)

    # 2) Reusar el image_processor (tokenizer) y compute_metrics que ya definiste
    # Asegúrate de tener id2label y image_processor igual que en el entrenamiento
    # Ejemplo:
    # id2label = {0: "background", 1: "Manzana", 2: "OtraClase"}
    filename = "id2label.json"
    with open(filename, "r", encoding="utf-8") as f:
        id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    image_processor = SegformerImageProcessor.from_pretrained(local_model_path)
    compute_metrics = build_compute_metrics(
        id2label=id2label,
        image_processor=image_processor,
        ignore_index=0,
        compute_precision=True
    )
    training_args = TrainingArguments(
        output_dir="out/segformer",
        eval_strategy="steps",   # o "epoch"
        eval_steps=500,                # ajusta al tamaño de tu dataset
        save_strategy="steps",         # guarda checkpoints sincronizados con eval
        save_steps=500,
        load_best_model_at_end=False,   # carga el mejor al final
        metric_for_best_model="mean_iou",  # puedes usar "mean_iou" si prefieres
        greater_is_better=True,        # True para f1_macro y mean_iou
        logging_steps=100,
        save_total_limit=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        seed=42,
    )

    from PIL import Image
    import numpy as np
    import torch
    from transformers import AutoImageProcessor

    checkpoint_path = r"D:\Github\Block-Detection\SegFormer\out\segformer\checkpoint-15"

    image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)

    ignore_index = 0  # ajusta si tu void/unlabeled es otro ID
    # Si tu unlabeled estaba en 255 en las máscaras originales, remapea a ignore_index

    def preprocess_example(example):
        # Ajusta los nombres de columnas de tu dataset:
        # ejemplo: example["image_path"] o example["image"] (PIL), example["mask_path"] o example["mask"]
        img = example["pixel_values"]
        if not isinstance(img, Image.Image):
            img = Image.open(img).convert("RGB")
        else:
            img = img.convert("RGB")

        mask = example["label"]
        if not isinstance(mask, Image.Image):
            mask = Image.open(mask).convert("L")
        else:
            mask = mask.convert("L")

        mask_np = np.array(mask, dtype=np.int64)

        # (Opcional) Remapeos de IDs si tu dataset usa 255 como void:
        # mask_np[mask_np == 255] = ignore_index

        # Usa el image_processor para crear tensores consistentes
        proc = image_processor(
            img, 
            segmentation_maps=mask_np, 
            return_tensors="pt"
        )
        # proc["pixel_values"] -> (1, C, H, W)
        # proc["labels"]       -> (1, H, W)

        return {
            "pixel_values": proc["pixel_values"].squeeze(0),  # (C, H, W)
            "labels": proc["labels"].squeeze(0).long(),       # (H, W), int64
        }

    # IMPORTANTE: elimina columnas originales para que el collator no intente colacionarlas
    val_ds_proc = val_ds.map(
        preprocess_example,
        remove_columns=val_ds.column_names
    )

    # 3) Crear el Trainer para evaluación
    trainer = Trainer(
        model=model,
        args=training_args,          # Puedes reusar los mismos argumentos o crear nuevos
        eval_dataset=val_ds_proc,        # Dataset de evaluación
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    # 4) Ejecutar evaluación
    metrics = trainer.evaluate()
elif mode=='val':
    pass