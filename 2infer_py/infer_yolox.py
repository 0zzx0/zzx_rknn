import cv2
import time
import numpy as np
from rknnlite.api import RKNNLite

from tools import preproc, demo_postprocess, multiclass_nms

# RKNN_MODEL = 'yolox_relu_decode.rknn'
RKNN_MODEL = '../1convert/yolox_relu_nodecode.rknn'
IMG = '1.jpg'
IMG_SIZE = (640, 640)
NMS_THR = 0.65
CON_THR = 0.45


def yolox_infer():
    # 初始化
    rknn_lite = RKNNLite()  
    # 加载模型
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    
    # 初始化运行环境 设置单npu核
    ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)

    # 加载图片
    ori_img = cv2.imread(IMG)

    t1 = time.perf_counter()
    ## 图像预处理
    img, ratio = preproc(ori_img, IMG_SIZE)
    # cv2.imwrite("a.jpg", img)

    # 推理
    outputs = rknn_lite.inference(inputs=[img[None, :, :, :]])

    # 后处理
    predictions = np.squeeze(outputs[0])
    predictions = demo_postprocess(predictions, IMG_SIZE)

    # nms
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
    boxes_xyxy /= ratio
    dets = multiclass_nms(boxes_xyxy, scores, NMS_THR, CON_THR)

    t2 = time.perf_counter()
    print(f"infer time: {(t2 - t1) * 1000}ms")

    print(f"infer ans: {dets}")

    # 销毁
    rknn_lite.release()

if __name__ == '__main__':
    yolox_infer()
