from PIL import Image
import cv2
from paddleocr import PaddleOCR, draw_ocr
import os
from string import punctuation
import copy
import pandas as pd
from OCRPytesseract import bag_of_words_accuracy
    


images=os.listdir("data/DS5110-ProjectSet/")
images=[image for image in images if ".jpg" in image]
all_tp=0
all_fn=0
all_fp=0
all_pp_tp=0
all_pp_fn=0
all_pp_fp=0
image_tp=[]
image_fn=[]
image_fp=[]
image_percision=[]
image_recall=[]
image_pp_tp=[]
image_pp_fn=[]
image_pp_fp=[]
image_pp_percision=[]
image_pp_recall=[]

ocr = PaddleOCR(lang='eng', use_angle_cls = True)

for ii, image_file in enumerate(images):
    print(f"{image_file:s} {ii+1:d}/{len(images):d}")
    with open("data/DS5110-ProjectSet/txt/"+image_file[:-4]+".txt",'r') as file:
        annotation=file.read()
    img=cv2.imread("data/DS5110-ProjectSet/"+image_file)
    result = ocr.ocr(img)
    prediction = str()
    for line in result[0]:
        prediction = prediction + line[1][0]

    true_positives, false_negatives, false_positives, found_words, pred_words_sub=bag_of_words_accuracy(prediction, annotation)
    print([true_positives, false_negatives, false_positives])
    image_tp.append(true_positives)
    image_fn.append(false_negatives)
    image_fp.append(false_positives)
    if true_positives==0 and false_positives==0:
        image_percision.append(float('nan'))
    else:
        image_percision.append(true_positives/(true_positives+false_positives))
    if true_positives==0 and false_negatives==0:
        image_recall.append(float('nan'))
    else:
        image_recall.append(true_positives/(true_positives+false_negatives))
    all_tp+=true_positives
    all_fn+=false_negatives
    all_fp+=false_positives
    
    
    pre_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pre_img = cv2.GaussianBlur(pre_img,(5,5),0)
    _,pre_img = cv2.threshold(pre_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    result = ocr.ocr(img)
    preprosses_pred = str()
    for line in result[0]:
        preprosses_pred = preprosses_pred + line[1][0]
    true_positives, false_negatives, false_positives, found_words, pred_words_sub=bag_of_words_accuracy(preprosses_pred, annotation)
    print([true_positives, false_negatives, false_positives])
    image_pp_tp.append(true_positives)
    image_pp_fn.append(false_negatives)
    image_pp_fp.append(false_positives)
    if true_positives==0 and false_positives==0:
        image_pp_percision.append(float('nan'))
    else:
        image_pp_percision.append(true_positives/(true_positives+false_positives))
    if true_positives==0 and false_negatives==0:
        image_pp_recall.append(float('nan'))
    else:
        image_pp_recall.append(true_positives/(true_positives+false_negatives))
    all_pp_tp+=true_positives
    all_pp_fn+=false_negatives
    all_pp_fp+=false_positives
    

out_df=pd.DataFrame.from_dict({"Image":images,"TruePositives":image_tp,"FalseNegatives":image_fn,"FalsePositives":image_fp,"Percision":image_percision,"Recall":image_recall,
                               "PreProcessTruePositives":image_pp_tp,"PreProcessFalseNegatives":image_pp_fn,"PreProcessFalsePositives":image_pp_fp,"PreProcessPercision":image_pp_percision,"PreProcessRecall":image_pp_recall})
out_df.to_csv("results/PaddleResults.csv")
print("\n")
print(f"Paddle Percision Without Pre Processing: {all_tp/(all_tp+all_fp):0.4f}")
print(f"Paddle Recall Without Pre Processing: {all_tp/(all_tp+all_fn):0.4f}")
print(f"Paddle Percision With Pre Processing: {all_pp_tp/(all_pp_tp+all_pp_fp):0.4f}")
print(f"Paddle Recall With Pre Processing: {all_pp_tp/(all_pp_tp+all_pp_fn):0.4f}")
