import numpy as np
import cv2
import sys
import cv2.bgsegm
import csv
from random import randint

fp = open('C:/Users/Eniac/Downloads/Curso_Subtraction-20231012T224441Z-001/Curso_Subtraction/videos/report.csv', mode='w', newline='')
writer = csv.DictWriter(fp, fieldnames=['Frame', 'Pixel Count'])
writer.writeheader()

TEXT_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
BORDER_COLOR = (randint(0, 255), randint(0, 255), randint(0, 255))
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 1.2
VIDEO_SOURCE = "C:/Users/Eniac/Downloads/Jupyter/Projetos/Curso_Subtraction/Resultados/TestesComparativos/people.mp4"
TITLE_TEXT_POSITION = (100, 40)

BGS_TYPES = ["GMG", "MOG", "MOG2", "KNN", "CNT"]

def getBGSubtractor(BGS_TYPE):
    if BGS_TYPE == "GMG":
       return cv2.bgsegm.createBackgroundSubtractorGMG()
    if BGS_TYPE == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if BGS_TYPE == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    if BGS_TYPE == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    if BGS_TYPE == "CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    print("Detector desconhecido")
    sys.exit(1)

cap = cv2.VideoCapture(VIDEO_SOURCE)

BGSubtractor = []
for i, a in enumerate(BGS_TYPES):
    BGSubtractor.append(getBGSubtractor(a))

def main():
    framecount = 0
    while cap.isOpened():
        ok, frame = cap.read()

        if not ok:
            print('Erro na captura')
            break

        framecount += 1
        frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)

        gmg = BGSubtractor[0].apply(frame)
        mog = BGSubtractor[1].apply(frame)
        mog2 = BGSubtractor[2].apply(frame)
        knn = BGSubtractor[3].apply(frame)
        cnt = BGSubtractor[4].apply(frame)

        gmgCount = np.count_nonzero(gmg)
        mogCount = np.count_nonzero(mog)
        mog2Count = np.count_nonzero(mog2)
        knnCount = np.count_nonzero(knn)
        cntCount = np.count_nonzero(cnt)

        writer.writerow({'Frame': 'MOG', 'Pixel Count': mogCount})
        writer.writerow({'Frame': 'MOG2', 'Pixel Count': mog2Count})
        writer.writerow({'Frame': 'GMG', 'Pixel Count': gmgCount})
        writer.writerow({'Frame': 'KNN', 'Pixel Count': knnCount})
        writer.writerow({'Frame': 'CNT', 'Pixel Count': cntCount})

        cv2.putText(frame, 'MOG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, 'MOG2', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, 'GMG', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, 'KNN', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)
        cv2.putText(frame, 'CNT', TITLE_TEXT_POSITION, FONT, TEXT_SIZE, TEXT_COLOR, 2, cv2.LINE_AA)

        cv2.imshow('Frame', frame)
        cv2.imshow('MOG', mog)
        cv2.imshow('MOG2', mog2)
        cv2.imshow('GMG', gmg)
        cv2.imshow('KNN', knn)
        cv2.imshow('CNT', cnt)

        cv2.moveWindow('Frame', 0, 0)
        cv2.moveWindow('MOG', 719, 0)
        cv2.moveWindow('KNN', 0, 250)
        cv2.moveWindow('GMG', 0, 250)
        cv2.moveWindow('MOG2', 719, 250)
        cv2.moveWindow('CNT', 719, 500)

        k = cv2.waitKey(1) & 0xff
        if k == 27: # ESC
            break

main()
