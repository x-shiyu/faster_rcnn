import torch
import cv2.cv2 as cv
import detectron2
import matplotlib.pyplot as plt
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from pathlib import Path
# from detectron2.utils.visualizer import ColorMode
setup_logger()

# import some common libraries
import numpy as np
import os, json, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode


def getBallonData(type):
    trainDataPath = os.path.abspath('./images/balloon/' + type + '/via_region_data.json')
    with open(trainDataPath) as f:
        imgInfos = json.load(f)
    imgDir = os.path.abspath('./images/balloon/' + type)
    allImages = []
    for index, item in enumerate(imgInfos.values()):
        record = {}
        filename = os.path.join(imgDir, item["filename"])
        height, width = cv.imread(filename).shape[:2]
        record["file_name"] = filename
        record["image_id"] = index
        record["height"] = height
        record["width"] = width
        annos = item["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        allImages.append(record)
    return allImages


# all_dicts = getBallonData()
for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: getBallonData(d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])


# balloon_metadata = MetadataCatalog.get("balloon_train")


def imgShow(img):
    plt.imshow(img)
    plt.show()


def bbxShow(pre, img) -> np.ndarray:
    pre_boxes = pre["instances"].pred_boxes
    pre_classes = pre["instances"].pred_classes.numpy()
    for index, item in enumerate(pre_boxes):
        if pre_classes[index] == 0:
            item = item.numpy()
            img = cv.rectangle(img, (item[0], item[1]), (item[2], item[3]), (0, 0, 0), 2)

    return img


def getReadyCfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("balloon_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)

    return cfg


def trainModel():
    cfg = getReadyCfg()
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def showRes():
    cfg = getReadyCfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = os.path.abspath("model-pussy.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set a custom testing threshold
    predictor = DefaultPredictor(cfg)
    imgDirPath = os.path.abspath('./images/test/182.png')
    img = cv.imread(imgDirPath)
    out = predictor(img)
    print(out)
    # for index, item in enumerate(DatasetCatalog.get('balloon_val')):
    # for index,item in enumerate(imgDirPath.iterdir()):
    #     imgPath = str(item)
    #     img = cv.imread(imgPath)
    #     out = predictor(img)
    #     res = bbxShow(out, img)
    #     cv.imwrite(os.path.abspath('./pre_out/' + str(index)+'_pussy.png'), res)


showRes()
