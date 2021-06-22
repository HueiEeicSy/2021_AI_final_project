# Instructions
## **Training Time**
:warning: **All the code should be run on Google Colab**
<details open>
<summary>Download Yolov5 From Github</summary>

- Download code from github: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
```bash
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!git reset --hard 886f1c03d839575afecb059accf74296fad395b6
```
</details>

<details open>
<summary>Prepare Environment</summary>

```bash
!pip install -qr requirements.txt  # install dependencies (ignore errors)
import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
```
</details>

<details open>
<summary>Download Dataset From Roboflow</summary>

- You could upload your own datasets or download our prepared one mentioned below

⭐ 636 and 683 means the original number of data
⭐ We add more confusing data in the 683 version


| Augmentation |                                                               |                                                               |                                                               |                                                               |                                                               |                                                               |                                                               |                                                               |
|:------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|:-------------------------------------------------------------:|
|    Rotate    |                                                               |                      :heavy_check_mark:                       |                                                               |                      :heavy_check_mark:                       |                      :heavy_check_mark:                       |                      :heavy_check_mark:                       |                      :heavy_check_mark:                       |                      :heavy_check_mark:                       |
|  Brightness  |                                                               |                                                               |                      :heavy_check_mark:                       |                      :heavy_check_mark:                       |                      :heavy_check_mark:                       |                      :heavy_check_mark:                       |                      :heavy_check_mark:                       |                      :heavy_check_mark:                       |
|     Flip     |                                                               |                                                               |                                                               |                                                               |                      :heavy_check_mark:                       |                                                               |                      :heavy_check_mark:                       |                      :heavy_check_mark:                       |
|     Blur     |                                                               |                                                               |                                                               |                                                               |                                                               |                      :heavy_check_mark:                       |                      :heavy_check_mark:                       |                                                               |
|     Gray     |                                                               |                                                               |                                                               |                                                               |                                                               |                                                               |                                                               |                      :heavy_check_mark:                       |
|  ***636***   | [link](https://app.roboflow.com/ds/2dKyOgouLP?key=6H6rOCIChI) | [link](https://app.roboflow.com/ds/jcIe9MDISG?key=IHyh9dFu85) | [link](https://app.roboflow.com/ds/V53TuCcgte?key=JIwulxBl3j) | [link](https://app.roboflow.com/ds/EwPN28c5Wj?key=SfIgF5Sly6) | [link](https://app.roboflow.com/ds/nOx1yXalf7?key=prHS29Bmgk) | [link](https://app.roboflow.com/ds/kpMoXRPWEw?key=pocQXmrxBn) | [link](https://app.roboflow.com/ds/kp9LcYLLPr?key=5Er4MRVC7H) | [link](https://app.roboflow.com/ds/NjHT1NaUBj?key=TUK86RDa3H) |
|  ***683***   | [link](https://app.roboflow.com/ds/pHtQPEQQmP?key=tL2vDPLpnm) |                                                               |                                                               | [link](https://app.roboflow.com/ds/nZq1hImwaZ?key=dPDqiQPGdR) | [link](https://app.roboflow.com/ds/6Tq4fVAjwa?key=EVVsbhbyQC) |                                                               |                                                               | [link](https://app.roboflow.com/ds/2nCYygNeab?key=qMFkktWLa1) |


```bash
%cd /content
!curl -L "<dataset download link>" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```
</details>

<details open>
<summary>Define Model Configuration and Architecture</summary>

- More details in *AI-Final-Project-Detect.ipynb*
</details>

<details open>
<summary>Train Custom YOLOv5 Detector</summary>

- Train with image size 416 *(default)*, batch 16 *(default)*, epochs 100 *(default)*

```bash
%%time
%cd /content/yolov5/img 416 --batch 16 --epochs 100 --data '../data.yaml' --cfg ./models/custom_yolov5s.yaml --weights '' --name yolov5s_results  --cache
```
</details>

<details open>
<summary>Evaluate Custom YOLOv5 Detector Performance</summary>

- Print stats from tensorboard or old school graphs

```bash
# Start tensorboard
%load_ext tensorboard
%tensorboard --logdir runs
```

```bash
# we can also output some older school graphs
from utils.plots import plot_results  # plot results.txt as results.png
Image(filename='/content/yolov5/runs/train/yolov5s_results/results.png', width=1000)  # view results.png
```

- Also display our ground truth data

```bash
# display our ground truth data
print("GROUND TRUTH TRAINING DATA:")
Image(filename='/content/yolov5/runs/train/yolov5s_results/test_batch0_labels.jpg', width=900)
```
</details>

<details open>
<summary>Export Trained Weights</summary>

- Export the trained weights to your own google drive

```bash
from google.colab import drive
drive.mount('/content/gdrive')
```

```bash
%cp /content/yolov5/runs/train/yolov5s_results/weights/best.pt /content/gdrive/My\ Drive
```
</details>

---

## **Detecting Time**
:warning: **All the code should be run on Google Colab**
<details open>
<summary>Download Yolov5 From Github</summary>

- Download code from github: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)
```bash
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!git reset --hard 886f1c03d839575afecb059accf74296fad395b6
```
</details>

<details open>
<summary>Prepare Environment</summary>

```bash
!pip install -qr requirements.txt  # install dependencies (ignore errors)
import torch
from IPython.display import Image, clear_output  # to display images
from utils.google_utils import gdrive_download  # to download models/datasets

print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))
```
</details>

<details open>
<summary>Testing Time</summary>

- Download trained weights and input images from your own google drive
```bash
from google.colab import drive
drive.mount('/content/gdrive')
```
- Set confidence threshold with 0.5 *(defalut)* and save the detecting result (including images with bounding boxes and txt files)
```bash
%cd /content/yolov5/
!python detect.py --weights ../gdrive/MyDrive/finalModel/best.pt --img 416 --conf 0.5 --save-txt --save-conf --source ../gdrive/MyDrive/testingImage/
```
- Calculate final outputs with more details in *AI-Final-Project-Detect.ipynb*
```bash
resultPath = glob.glob('/content/yolov5/runs/detect/exp/labels/*.txt') # may have to change the path
imagePath = glob.glob('/content/yolov5/runs/detect/exp/*.jpg') # may have to change the path
calculation(imagePath)
```
</details>