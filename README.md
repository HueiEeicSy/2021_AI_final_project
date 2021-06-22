# Coin Detection and Calculation

## Github repo link
https://github.com/HueiEeicSy/2021_AI_final_project

## Introduction

- 當我們身邊有很多零錢時，很難快速用肉眼去判斷這些零錢總共有多少，因此本次作業的目的就是希望使用者能透過簡單地方法，可以快速的計算錢幣的總價值。我們希望在光源充足的情況下，拍一張含有許多硬幣的照片，接著再透過我們訓練好的模型來快速辨認及計算照片中所包含多少錢，且盡量提升其精確度。
- 我們預想使用 Sliding Window、Faster-RCNN、YOLO 等 Object Detection 的技術去實現我們的目標，並且再透過產生出的 Bounding Box 及 Classification 的結果去計算錢幣的總價值。

![](https://i.imgur.com/1J38LLm.png)
Fig 1. expected model 

## Related work

在我們的 Coin Detection and Calculation 中，最核心的技術是做硬幣的偵測，也就是Object Detection的部分。

- **Object Detection technology**:
    
    在 Object Detection 中，主要分成兩個步驟:物件分類 (classfication) 以及物件位置檢測 (localization)，其中以位置檢測為最主要技術。
    - Sliding window 是最原始、暴力的作法，設定不同尺寸的 windows，由影像左上開始滑動，列出所有區域，並將其作 classfication 查看何者符合物件的機率最高，即為結果。但 sliding window 的作法太過耗時，要經歷過太多種 windows，因此後續有了 Region Proposals 的方法。
    - Region Proposals 主要是利用影像中的紋理、邊緣或顏色等資訊，定義出物體可能出現的位置，再利用 classfication 對其分類，此一方法相較 sliding windows 大量降低了 windows 數量，相關做法有 R-CNN、Fast R-CNN...等。
    - 另外的方法是 Regression，利用將影像切成 grid 的方式，在以 grid 中心來選擇不同 size 的 window，並用 regrssion 的方法計算含有 object 的機率，相關做法有 YOLO、SSD...等。

- **Coin Detection from others**:
    
    對於硬幣偵測的問題，在[4]中，利用影像處理技巧先得到原始影像的 edge，再產生 reference circles (radius 從小到大)去滑過整個 edge的 image 做比對，以偵測該物體是否為硬幣，該方法類似 sliding window，且無對偵測出的硬幣做 classfication。
    而在[5]中,先運用 Circular Hough Transform (CHT)偵測 circular objects，CHT類似[4]中的窮舉，會花費大量時間，因此用downsample 將 coin size 限制在特定範圍內，減少偵測時間。而在classfication 中，運用 coin 的 relative size及Fisher Linear Discriminant Analysis (LDA)得出分類結果。

- **Our method**:

    我們的方法則是用 YOLO 當主要架構，希望透過其 regression 的方法在 object localization 上減少時間，而在得到 objects 後則用 CNN 來學習不同硬幣的特徵，並對其分類。
    另外由於硬幣會有多種轉向、亮暗、反光，因此我們在 dataset 上面嘗試了多種 augmentation 讓 model 能學到更多樣的特徵。

## Methodology
![sdfef123456sdfsfes](https://i.imgur.com/ntd5ZFp.png)
Fig 2. Pipeline of our model
- **Datasets**
    - 首先要自己拍各種1、5、10、50元硬幣，class也是分為這四種，並用 [*labelImg*](https://github.com/tzutalin/labelImg) 手動標記資料位置及分類。
    - 將資料送至 [*roboflow*](https://docs.roboflow.com/image-transformations/image-augmentation) 做 Data Augmentation。
    此外，有約1500張的資料拿來做 training，約100張拿來做 validation，最後106張用於 testing。
- **Data Augmentation**: 
    - 由於本次作業的 Data 都是由我們自己準備，因此在 Data 的數量上會有一定的劣勢。為解決這樣的問題，我們採取 Data Augmentation 的方法來增加我們 Data 的數量，且也能試著讓我們的 model 學到更多的特徵，使得最終訓練過後的 model 可以有更好的預測精確度。
    - 在本次作業中，我們利用 Roboflow 網站來進行 Data Augmentation，而採用的方法有以下幾種：
        - Rotation: 將照片旋轉90度、180度或270度，適用於那些方向並不重要且不固定的照片。
        - Brightness: 將照片的亮度提升或降低20%。
        - Flip: 將照片上下或左右翻轉。
        - Crop: 隨機針對局部照片做裁減，可針對那些只出現在畫面一半的物體做加強。
        - Blur: 將照片做一定程度的模糊處理。
        - Grayscale: 將照片轉為灰階。

    ![](https://i.imgur.com/MrcLF0T.png)
Fig 3. Different augmentation

訓練與測試主要從ultraytics的[*yolov5*](https://github.com/ultralytics/yolov5)與其[colab版本](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)修改。
- **Model Configuration**

    我們採用 yolov5s.yaml 當作模板。

    其 depth_multiple=0.33 與 width_multiple=0.5 代表模型的深度與kernel(卷積核)都較小。
    接下來是 yolov5 的架構。
    ![](https://pic1.zhimg.com/80/v2-770a51ddf78b084affff948bb522b6c0_720w.jpg)

    - input
    yolov5 與 yolov4 一樣採用 Mosaic Mosaic Augmentation，將資料隨機縮放、裁減提升小目標的訓練效率。此外還有自適應的 anchor 計算(anchor box與ground truth的box比對後自動更新)與縮放圖片(非正方形的圖片會自動加上黑邊變成416\*416)。
    - backbone
    ![](https://pic3.zhimg.com/80/v2-139a50003c09efe54b2db906710f6252_720w.jpg)
    首先會進入 yolov5 特有的 Focus 層，將圖片(608\*608\*3)切成更小的feature map(304\*304\*32)。之後是三層 yolov4 引進的CSP層，最後將它們切成19\*19，增加學習效率與所需記憶體。
    - head
    ![](https://pic4.zhimg.com/80/v2-d8d0ff4768a92c9a10adbe08241c0507_720w.jpg)
    yolov5 與 yolov4 一樣採用FPN+PAN的設計，也就是將19\*19的圖片往上採樣到76\*76(FPN)，之後再往下採樣到19\*19(PAN)。
    - prediction
    yolov5 採用 GIOU_loss 當作 loss function，與傳統的差在即使兩個box的IOU相同，也可以分辨它們與 ground truth 的距離遠近。
    
    

- **Training & Testing**
    - 在 training 過程中，圖片解析度設為416*416以提升訓練速度。Batch size 為16，也就是每步訓練16筆資料。Epochs 首先設為100，最終再以較好的幾組實驗進行更長時間的訓練，此時會將 eopochs 設為300，也就是將全部資料訓練共300次。最後選擇 loss 最低的 weight作為我們最終的 model。而訓練的數據可從 tensorboard 觀看，避免 overfitting。
    - testing 過程則是利用106張不同環境的測試照片，並以此 [*repo*](https://github.com/rafaelpadilla/Object-Detection-Metrics) 計算其mAP，藉此評斷訓練成果，像是資料量是否足夠或哪種 augmentation 的組合較佳。
    
- **Final Prediction**
    - Confidence Threshold: 針對某一個分類需要多少的 Confidence 我們也有設一個下界，根據我們的比較，我們選擇0.5作為我們 Confidence 的門檻，使得 Model 能偵測到更多硬幣的情況下，又不會在背景的部分錯誤偵測到硬幣。
    - Overlapping Bounding Boxes: 由於在本次作業中較可能出現一個物體卻同時被分配到多種分類的情況，例如照片中的10元硬幣同時被分類到5元及10元的類別，因此我們採取的作法是根據 Bounding box 在圖片上位置的相似度來決定是否達成重疊的條件。詳細過程為，利用 Bounding box 在圖片中X軸及Y軸中心的值進行兩兩比對，若兩者皆小於0.05，我們就會判定他為重疊的 Bounding box，而只採用 Confidence 較高的那個 Bounding box 及分類。
   
- **Final Output**
    - Total value of coins: 利用偵測結果的分類去計算同一圖片內全部硬幣的價值，並顯示於圖片中。



## Experiments

- **Data Augmentation**:
    我們針對各種影像處理手段進行了不同組合的嘗試，結果發現了Roate、Brightness、Flip、Blur的組合能將model對testing data的mAP提升最多，而Grayscale、Crop等則沒有太大效用。
    
Table 1. Performance of different augmentation.
![](https://i.imgur.com/LXZEiqw.png)



- **Dataset**
    而從Table 1可以發現，經過augmentation雖然有提升mAP，但對5塊硬幣的分類能力明顯偏低，推測是因為5、10塊的顏色類似以及1、5塊人臉面十分相似，從而導致5塊的分類結果不理想。
    因此我們額外在training data增加(5,10)和(1,5)等容易判別錯誤的pair，也另外加深訓練的epoch、用pretrained weight訓練，來改善此問題。
    
    Table 2. Performance with/without ambiguous data and different epoch.

    ![](https://i.imgur.com/0JgUuvw.png)

    可以發現經過training data、epoch的加強及加入pretrained weight後，除了mAP提升，5塊錢的AP更是有顯著的進步。
    

- **Result**
![](https://imgur.com/CrjlZRo.jpg =480x360)
上圖是我們的最終結果，方框是預測結果、左下角是硬幣的總和。 

   

| original model | final model |
| -------- | ---------|
|  ![](https://i.imgur.com/jwEAxvw.jpg)|![](https://i.imgur.com/mLqLTmU.jpg)|
|   ![](https://i.imgur.com/UqontNe.jpg)| ![](https://i.imgur.com/kY6toXq.jpg)|
 上方表格左側是最原始的model(沒做augmentation、沒加ambiguous data)的預測結果，右側則是我們最終的model的預測結果。(有打紅色叉的是錯誤的結果)
 可發現左側分類能力不佳(尤其是5塊)，而右側整體分類能力提升，且更能正確偵測出5塊。
 
## Conclusion
我們這次的 project 利用了 Object detection 的概念解決數硬幣的問題，以 Yolov5s 當作 model 的 backbone，也透過對 data 不同 augmentation 的嘗試、組合以及 epch、weight 的調整，來增加整體硬幣偵測、識別的準確度。而最終所 train 出來的 model 也確實能在 coin detection 上有不錯的功效，達成幫助我們算錢的目的。

## References
- [1] YOLO-v5 https://github.com/ultralytics/yolov5
- [2] Colab https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/
- [3] Datasets Processing https://app.roboflow.com/
- [4] Coin detection: Discovering OpenCV with Python https://dev.to/tinazhouhui/coin-detection-discovering-opencv-with-python-1ka1
- [5] Automatic Coin and Bill Detection Dominic Delgado,Stanford University, 650 Serra Mall, Stanford, CA 94305, delgado4@stanford.edu
- [6] labelImg https://github.com/tzutalin/labelImg
- [7] calculate mAP https://github.com/rafaelpadilla/Object-Detection-Metrics
- [8] yolov5 explaination https://zhuanlan.zhihu.com/p/172121380?fbclid=IwAR1EHZQzecP30Ytr5vgF-lkikTOGeOTgOK6hc3DEC_5olU60H_KQIQWXzw0
