import cv2
import numpy as np 
import time
from datetime import datetime
import json

model_path = '../model/haarcascade_frontalface_default.xml'
config_path = "../config_test.json"

class collect(object):
    def __init__(self):
        self.get_photo()
    def get_photo(self):
        detector = cv2.CascadeClassifier(model_path)  # 載入人臉追蹤模型
        recog = cv2.face.LBPHFaceRecognizer_create()      # 啟用訓練人臉模型方法
        faces = []   # 儲存人臉位置大小的串列
        ids = []     # 記錄該人臉 id 的串列

        with open(config_path) as j:
            config = json.load(j)
            len_id = config['len']

        times = 1 #這個參數決定要訓練幾個人臉

        for i in range(1,times+1):
            sec = 0
            print('camera...')                     # 提示啟用相機
            cap = cv2.VideoCapture(0)                # 啟用相機
            if not cap.isOpened():
                print("Cannot open camera")
                exit()
            while True:
                ret, img = cap.read()             # 讀取影片的每一幀
                if not ret:
                    print("Cannot receive frame")
                    break
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 色彩轉換成黑白
                img_np = np.array(gray,'uint8')         # 轉換成指定編碼的 numpy 陣列
                face = detector.detectMultiScale(gray)      # 擷取人臉區域
                for(x,y,w,h) in face:
                    faces.append(img_np[y:y+h,x:x+w])     # 記錄自己人臉的位置和大小內像素的數值
                    ids.append(int(i+int(len_id)))               # 記錄自己人臉對應的 id，只能是整數，都是 1 表示川普的 id
                    print(int(i+int(len_id)))
                cv2.imshow('vivi', img)             # 顯示攝影機畫面
                # if cv2.waitKey(100) == ord('q'):        # 每一毫秒更新一次，直到按下 q 結束
                cv2.waitKey(100)
                sec+=0.1
                if sec>=5:    #拍攝五秒後關閉
                    break

            print('training...')                        # 提示開始訓練
            print('ok!')
            time.sleep(5)
        recog.train(faces,np.array(ids))                  # 開始訓練

        if config['init'] == "True":
            recog.save(f'../data/face_model_main.yml')
            recog.save(f'../data/face_model_{str(i+int(len_id))}.yml')                       # 訓練完成儲存為 face.yml  
            # recog.save(f'../data/face_model_test.yml')                                          # 測試用
            # recog.save(f'../data/face_model_{str(i+int(len_id))}_test.yml')                     # 測試用
            with open(config_path) as j:
                config = json.load(j)
                dict_ = config
                dict_['init'] = "False"
                dict_['len'] = str(i+int(len_id))
            with open(config_path, 'w') as j:
                json.dump(dict_, j)

        else:
            recog.save(f'../data/face_model_{str(i+int(len_id))}.yml')                       # 訓練完成儲存為 face.yml
            # recog.save(f'../data/face_model_{str(i+int(len_id))}_test.yml')                       # 測試用
            with open(config_path) as j:
                config = json.load(j)
                dict_ = config
                dict_['len'] = str(i+int(len_id))
            with open(config_path, 'w') as j:
                json.dump(dict_, j)
if __name__ == "__main__":
    c = collect()