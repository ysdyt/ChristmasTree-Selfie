# -*- coding: utf-8 -*-
import cv2
import sys
import logging as log
import datetime as dt
import time
from time import sleep
import subprocess

# RaspberryPi用GPIOピン制御のモジュール
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(11,GPIO.OUT)

# 顔検出に使う特徴量ファイルの指定
cascPath = '/home/pi/face_detect/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

# ログの出力
log.basicConfig(filename='webcam.log',level=log.INFO)

# もろもろ初期化
video_capture = cv2.VideoCapture(0)
anterior = 0
shot_dense = 0.5
considerable_frames = 20
prev_faces = []
prev_shot = None


while True:
    # 動画の読み込みに失敗した場合
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # 動画の読み込みに成功した場合
    # 読み込んだ動画を1フレームづつ読み込む
    ret, org_frame = video_capture.read()
    
    shape = org_frame.shape
    # 画像サイズを小さくするための割合を設定（後述:詳細1）
    ratio = 3

    # 処理高速化のために画像サイズを小さくする（後述:詳細1）
    frame = cv2.resize(org_frame, (shape[1]/ratio,shape[0]/ratio))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 顔の検出
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        #画像サイズの小さくしたので走査するwindowも同様の割合で小さくする（後述:詳細1）
        minSize=(30/ratio, 30/ratio)
    )
    print prev_faces, prev_shot

    # 顔が検出されるたびに連続で撮影されないための工夫（後述:詳細2）
    # prev_shot is None -> 初回のデータ格納   seconds>3 -> 顔検出不感タイムの指定（2週目以降の処理）
    if prev_shot is None or (dt.datetime.now() - prev_shot).seconds > 3:
        prev_faces.append(len(faces))
        if len(prev_faces) > considerable_frames:
            drops = len(prev_faces) - considerable_frames
            # prev_facesのリストの先頭からdrops分だけ要素を削除したlistを返す（20を超えた要素で古いものから削除していく）
            # prev_facesは常に20の要素数を保つ
            prev_faces = prev_faces[drops:]
        # その20の要素数のうち、0以上の数が全要素数(=20)のどれくらいの割合を占めるかチェック
        dense = sum([1 for i in prev_faces if i > 0]) / float(len(prev_faces))

        # prev_facesに20よりも多くの要素が格納されようとしたとき
        if len(prev_faces) >= considerable_frames and dense >= shot_dense:
            print 'shot',str(dt.datetime.now())
            save_fig_name = '/home/pi/face_detect/save_fig/{}.jpg'.format(dt.datetime.now())
            cv2.imwrite(save_fig_name,org_frame)
            
            # システムコマンドでConfluence APIを叩いて指定のページに保存した画像を投稿する（後述:詳細3）
            curl_cmd = "curl -D- -u ユーザ名:ドメイン名# -X POST -H \"X-Atlassian-Token: nocheck\" -F \"file=@{0}\"\ \"https://ドメイン.atlassian.net/wiki/rest/api/content/コンテンツID/child/attachment\"".format(save_fig_name)
            subprocess.call(curl_cmd,shell=True)

            # 画像がAPIに渡されたら目印としてラズパイにくっつけたLEDを明滅させる（後述:詳細4）
            # ここでは適当に3回分明滅させる
            n = 0
            while n <3:
                GPIO.output(11,True)
                time.sleep(0.3)
                GPIO.output(11,False)
                time.sleep(0.3)
                n += 1
                
            prev_faces = []
            prev_shot = dt.datetime.now()

    # 顔検出領域を四角で囲む
    for (x, y, w, h) in faces:
        x_ = x*ratio
        y_ = y*ratio
        x_w_ = (x+w)*ratio
        y_h_ = (y+h)*ratio
        cv2.rectangle(org_frame, (x_, y_), (x_w_, y_h_), (0, 255, 0), 2)

    # 'q'を押したら検出処理を終える
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 接続しているモニターにリアルタイムで画像を描写する（カメラで撮っている動画を表示する）
    cv2.imshow('Video', org_frame)

# 全てが完了したらプロセスを終了
video_capture.release()
cv2.destroyAllWindows()
