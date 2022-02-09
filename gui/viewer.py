#GUI 모듈
from tkinter import *
import tkinter.ttk as ttk

#파일처리 모듈
from tkinter import filedialog
from PIL import ImageTk,Image
import sys
import os
import time


#yolo detect 관련 모듈
import cv2
#from model import detect
import threading


######################################
# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage - sources:
    $ python path/to/detect.py --weights yolov5s.pt --source 0              # webcam
                                                             img.jpg        # image
                                                             vid.mp4        # video
                                                             path/          # directory
                                                             path/*.jpg     # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python path/to/detect.py --weights yolov5s.pt                 # PyTorch
                                         yolov5s.torchscript        # TorchScript
                                         yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                         yolov5s.xml                # OpenVINO
                                         yolov5s.engine             # TensorRT
                                         yolov5s.mlmodel            # CoreML (MacOS-only)
                                         yolov5s_saved_model        # TensorFlow SavedModel
                                         yolov5s.pb                 # TensorFlow GraphDef
                                         yolov5s.tflite             # TensorFlow Lite
                                         yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)




######################################

detect_weights = 'model\\runs\\train\\6_class_final\\weights\\best.pt'
result_img = 0



root = Tk() 
#gui이름
root.title("gui test project")

#파일 오픈 , 종료  버튼 만들기n
file_button_frame = Frame(root)
file_button_frame.pack(fill="x", padx=5, pady=5)


def file_open_fun():
  files = filedialog.askopenfilenames(title= "이미지 파일을 선택하세요",\
    filetypes= (("PNG 파일", "*.png"), ("모든 파일", "*.*")),
    #최초에 보여줄 dir를 명시
    #r을 쓰면 넣어준 문자그대로를 경로로 사용
    initialdir=r"D:\파이선GUI공부\guiproject\예시사진"
    )
    #사용자가 선택한 파일목록 출력

  
  for file in files:
    #이제 리스트박스에 출력을 해주면됨
    list_file.insert(END,file)
    
    run(weights=[detect_weights] ,  # model.pt path(s)
        source= str(file) ,  # file/dir/URL/glob, 0 for webcam
        data=''  , # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=True,  # show results 수정
        save_txt=False,  # save results to *.txt
        save_conf=True,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project='runs\detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        )





file_open_button = Button(file_button_frame,text="파일열기", padx=5, pady=5, width=10, command=file_open_fun)
file_open_button.pack(side="left")

file_exit_button = Button(file_button_frame,text="종료", padx=5, pady=5, width=10, command=root.quit)
file_exit_button.pack(side="right")




#------------------------------------------------------------------------------------------------------------------

#2-1
#오픈된 파일 경로 표시 

#list프레임 그리고 스크롤바
list_frame = Frame(root)
#화면 전체에 펴지도록 프레임을 both로 채움
list_frame.pack(fill="both", padx=5, pady=5)

#스크롤바 구현
scrollbar= Scrollbar(list_frame)
#스크롤바는 리스트 프레임 오른쪽에 그리고 y축으로 쭉 늘리기
scrollbar.pack(side="right", fill="y")

#리스트박스를 실제 구현
#리스트 프레임 넣고, 높이는 15면 한번에 15개의 파일을 보고, 스크롤바와 연동하기 위해 yscrollcommand=scrollbar.set으로 스크롤바와 매핑-1
list_file = Listbox(list_frame, selectmode="extended", height =15, yscrollcommand=scrollbar.set)
list_file.pack(side="left",fill ="both", expand=True)

#스크롤바 -> 리스트파일과 mapping-2
scrollbar.config(command=list_file.yview)


#------------------------------------------------------------------------------------------------------------------

#파일 경로 저장 배열

file_path_arr=[]


#3-1선택한 파일 경로 넘기기 버튼 

#3-2선택된 파일 삭제 버튼


select_file_button_frame = Frame(root)
select_file_button_frame.pack(fill="both", padx=5, pady=5)


#3-1 선택한 파일 경로 넘기기 버튼 함수

def file_send_func():
  for index in list_file.curselection():
    #print(list_file.get(index))
    file_path_arr.append(list_file.get(index))



    

#3-2 선택된 파일 삭제 버튼 함수
def file_del_func():

  #print(list_file.curselection())
  #보통 삭제시 앞에서부터 지우게되면 index가 하나씩 앞으로 당겨지므로 뒤에서부터 지움
  #reverse()는 리스트를 아예바꿈
  #revesed()는 바뀐 리스트를 반환
  for index in reversed(list_file.curselection()):
    list_file.delete(index)

#3-3 확인용
def file_path():
  for elem in file_path_arr:
    print(elem)

    

#3-1선택한 파일 경로 넘기기 버튼 

#send button
file_send_button = Button(select_file_button_frame,text="파일 경로 보내기", padx=5, pady=5, width=15, command=file_send_func)
file_send_button.pack(side="left")


#3-2선택된 파일 삭제 버튼
file_del_button = Button(select_file_button_frame,text="선택 삭제", padx=5, pady=5, width=15, command=file_del_func)
file_del_button.pack(side="right")



#3-3 경로저장 확인 
file_path_button = Button(select_file_button_frame,text="경로 확인", padx=5, pady=5, width=15, command=file_path)
file_path_button.pack(side="right")



#------------------------------------------------------------------------------------------------------------------

#4-1 선택된 사진 보기 
photo_frame = Frame(root)
photo_frame.pack(fill="x", padx=5, pady=5)

def photo_open():
  
  my_img = PhotoImage(file = "D:\파이선GUI공부/check.png",width=10,height=10)
  my_label = Label(root, image=my_img)
  my_label.pack(expand=1, anchor=CENTER)

  print(my_img)





photo_open_button = Button(photo_frame,text="사진 보기", padx=5, pady=5, width=10, command=photo_open)
photo_open_button.pack(side="left")


# photo_label_frame= LabelFrame(root, text="사진")
# photo_label_frame.pack(fill="x", padx=100, pady=100,ipady=200)








root.geometry("1600x1024")
root.mainloop()