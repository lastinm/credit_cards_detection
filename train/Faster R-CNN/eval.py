from utils.dataset import CocoDataset
import torch
from utils.model_utils import InferFasterRCNN,display_gt_pred
from pycocotools.coco import COCO
import os
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import json
import gc
import numpy as np

def save_json(data, file_path):
    """Сохранение данных в JSON с обработкой numpy и torch типов"""
    def convert(o):
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.int32, np.int64)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return o.cpu().numpy().tolist()
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
    with open(file_path, 'w') as file:
        json.dump(data, file, default=convert)

def evaluate_model(image_dir,
                   gt_ann_file,
                   model_weight,
                   device="cuda"):
    
    # 1. Инициализация COCO для ground truth
    if not os.path.exists(gt_ann_file):
        raise FileNotFoundError(f"GT annotations file not found: {gt_ann_file}")

    _ds = CocoDataset(
            image_folder=image_dir,
            annotations_file=gt_ann_file,
            height=640,
            width=640,
        )
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    IF_C = InferFasterRCNN(num_classes=_ds.get_total_classes_count() + 1,
                        classnames=_ds.get_classnames())

    IF_C.load_model(checkpoint=model_weight,
                    device=device)

    image_dir = image_dir

    cocoGt=COCO(annotation_file=gt_ann_file)
    imgIds = cocoGt.getImgIds() # all image ids

    res_id = 1
    res_all = []
        
    for id in tqdm(imgIds,total=len(imgIds)):
        id = id
        img_info = cocoGt.loadImgs(imgIds[id])[0]
        annIds = cocoGt.getAnnIds(imgIds=img_info['id'])
        ann_info = cocoGt.loadAnns(annIds)
        image_path = os.path.join(image_dir, 
                                img_info['file_name'])
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found {image_path}")
            continue

        transform_info = CocoDataset.transform_image_for_inference(image_path,width=640,height=640)
        result = IF_C.infer_image(transform_info=transform_info,
                                visualize=False)

        if len(result)>0:
            pred_boxes_xyxy = result['unscaled_boxes']
            pred_boxes_xywh = [[i[0],i[1],i[2]-i[0],i[3]-i[1]] for i in pred_boxes_xyxy]
            pred_classes = result['pred_classes']
            pred_scores = result['scores']
            pred_labels = result['labels']

            for i in range(len(pred_boxes_xywh)):
                res_temp = {"id":res_id,
                            "image_id":id,
                            "bbox":pred_boxes_xywh[i],
                            "segmentation":[],
                            "iscrowd": 0,
                            "category_id": int(pred_labels[i]),
                            "area":pred_boxes_xywh[i][2]*pred_boxes_xywh[i][3],
                            "score": float(pred_scores[i])}
                res_all.append(res_temp)
                res_id+=1

    # 3. Проверка наличия детекций
        else:
            print("Warning: No detections above confidence threshold!")
            # Возвращаем нулевые метрики
            return {'AP_50_95':0,
                    'AP_50':0}

    save_json_path = 'test_dect.json'
    save_json(res_all,save_json_path)

    # 4. Проверка файла перед загрузкой
    if not os.path.exists(save_json_path) or os.path.getsize(save_json_path) == 0:
        raise ValueError("Detection results file is empty or not created")
    
    cocoGt=COCO(gt_ann_file)
    #{{LMA}} cocoDt=cocoGt.loadRes(save_json_path)

    try:
        with open(save_json_path, 'r') as f:
            detections = json.load(f)
            if not detections:
                raise ValueError("Empty detections in JSON file")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in detections file")
    
    # 5. Загрузка результатов в COCO
    try:
        cocoDt = cocoGt.loadRes(save_json_path)
    except IndexError as e:
        print("Error loading detections. Possible causes:")
        print("- No valid detections in any image")
        print("- Invalid detection format")
        print(f"Debug: First 5 detections: {detections[:5]}")
        raise

    # 5. Загрузка результатов в COCO
    try:
        cocoDt = cocoGt.loadRes(save_json_path)
    except IndexError as e:
        print("Error loading detections. Possible causes:")
        print("- No valid detections in any image")
        print("- Invalid detection format")
        print(f"Debug: First 5 detections: {detections[:5]}")
        raise

    cocoEval = COCOeval(cocoGt,cocoDt,iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    
    AP_50_95 = cocoEval.stats.tolist()[0]
    AP_50 = cocoEval.stats.tolist()[1]
    
    del IF_C,_ds
    os.remove(save_json_path)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    return {'AP_50_95':AP_50_95,
            'AP_50':AP_50}