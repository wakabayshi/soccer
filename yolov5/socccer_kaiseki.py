#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 13:18:31 2021

@author: wakabayashi
"""

#########座標解析##############################################
from pathlib import Path
# rgbの数値を返す関数
def rgb(color):
    data = [["white", 255, 255, 255],
            ["black", 0, 0, 0],
            ["blue", 0, 0, 255],
            ["red", 255, 0, 0],
            ["grey", 190, 190, 190],
            ["LightGray", 211, 211, 211],
            ["green", 0, 255, 0],
            ["orange", 255, 165, 0],
            ["yellow", 255, 255, 0],
            ["pink", 255, 192, 203],
            ["snow", 255, 250, 250],
            ["cyan", 0, 255, 255],
            ["gold", 255, 215, 0],
            ["ivory", 255, 255, 240],
            ]
    for i in range(len(data)):
        if data[i][0].lower() == color.lower():
            return data[i][1], data[i][2], data[i][3]
#############################################################################################################        
#### 入力パラメタ設定 ############################
## （０）　ワークディレクトリ　
workdir =str(Path.home())+'/ildoonet-tf-pose-estimation/画像解析/成功例/'
file_head = '成功例_mp4_cmu_result'
#file_head = '/松山英樹側面抜粋'
## （１）　img, jsonディレクトリ内の解析したい範囲（連番）の開始番号、終了番号を指示。 ファイルがあれば上書きするので注意。
img_start = 2
img_end = 77
parts_set = ['LWrist','RWrist','LAnkle','RAnkle']
color_set = [rgb('red'),rgb('blue'),rgb('green'),rgb('pink')]
'''
#POSE_COCO_BODY_PARTS 
pairs = {{0,  "Nose"}, {1,  "Neck"}, {2,  "RShoulder"}, {3,  "RElbow"}, {4,  "RWrist"},\
        {5,  "LShoulder"}, {6,  "LElbow"}, {7,  "LWrist"}, {8,  "RHip"}, {9,  "RKnee"},\
        {10, "RAnkle"}, {11, "LHip"},  {12, "LKnee"}, {13, "LAnkle"}, {14, "REye"},\
        {15, "LEye"}, {16, "REar"}, {17, "LEar"}, {18, "Bkg"}}

'''
#### 入力パラメタ ############################
############################################################################################################# 
import pandas as pd
import json
import numpy as np
import cv2
from statistics import mean
import matplotlib.pyplot as plt
import os
import seaborn as sns

import time
os.chdir(workdir)
########################################
# image_file_dir0 =  file_head+'_img'
json_file_dir =  file_head+'_json/'
# json_xlsx_output = file_head+'_Json2xlsx/'
# ########################################
# image_file_dir =  image_file_dir0+'/'
# image_file_outdir =  image_file_dir0+'_tracing/'
# if not os.path.exists(workdir+image_file_outdir):#ディレクトリがなかったら
#     os.mkdir(workdir+image_file_outdir)#作成したいフォルダ名を作成
# if not os.path.exists(workdir+json_xlsx_output):#ディレクトリがなかったら
#     os.mkdir(workdir+json_xlsx_output)#作成したいフォルダ名を作成
#    print('dir='+image_file_outdir+' was newly made.')
##注意
#start_img = 157
#########################################################################################################

POSE_COCO_PAIRS = [	(0,1), (0,14),	(0,15),	(14,16),	(15,17), \
                (1,2),(1,5),(2,3),	(3,4),	(5,6),	(6,7),\
                (8,9),	(9,10),   (11,12),	(12,13),\
                (1,8),   (1,11),\
                (1,14),	(1,15),	(14,16),	(15,17)]

'''
    POSE_COCO_PAIRS = [	(0,  4),	(4,  5),	(6,  7),	(7,  8),	(9,  10),\
	(10, 11),	(12, 13),	(13, 14),	(1,  2),	(2,  9),\
	(2,  12),	(2,  3),	(2,  6),	(3,  17),\
	(1,  16),	(1,  17),	(16, 17)]
'''    
def gravityMean2(a,b):
    if a!=0.0 and b!=0.0:
        return mean([a,b])
    else:
        return 0
def graviryMean3(a,b,c):
    if a!=0.0 and b!=0.0 and c!=0.0:
        return mean([a,b,c])
    else:
        return 0
def calc_CoG(df):
    mass_elements.loc['head','X'] = df.loc['Nose','X'];mass_elements.loc['head','Y'] = df.loc['Nose','Y']
    mass_elements.loc['body','X'] = graviryMean3(df.loc['Neck','X'],df.loc['RHip','X'],df.loc['LHip','X'])
    mass_elements.loc['body','Y'] = graviryMean3(df.loc['Neck','Y'],df.loc['RHip','Y'],df.loc['LHip','Y'])
    #
    mass_elements.loc['R_upperArm','X'] = gravityMean2(df.loc['RShoulder','X'],df.loc['RElbow','X'])
    mass_elements.loc['R_upperArm','Y'] = gravityMean2(df.loc['RShoulder','Y'],df.loc['RElbow','Y'])
    mass_elements.loc['R_lowerArm','X'] = gravityMean2(df.loc['RElbow','X'],df.loc['RWrist','X'])
    mass_elements.loc['R_lowerArm','Y'] = gravityMean2(df.loc['RElbow','Y'],df.loc['RWrist','Y'])
    mass_elements.loc['L_upperArm','X'] = gravityMean2(df.loc['LShoulder','X'],df.loc['LElbow','X'])
    mass_elements.loc['L_upperArm','Y'] = gravityMean2(df.loc['LShoulder','Y'],df.loc['LElbow','Y'])
    mass_elements.loc['L_lowerArm','X'] = gravityMean2(df.loc['LElbow','X'],df.loc['LWrist','X'])
    mass_elements.loc['L_lowerArm','Y'] = gravityMean2(df.loc['LElbow','Y'],df.loc['LWrist','Y'])
    mass_elements.loc['R_parm','X'] = df.loc['RWrist','X']
    mass_elements.loc['R_parm','Y'] = df.loc['RWrist','Y']
    mass_elements.loc['L_parm','X'] = df.loc['LWrist','X']
    mass_elements.loc['L_parm','Y'] = df.loc['LWrist','Y']
    #
    mass_elements.loc['R_upperLeg','X'] = gravityMean2(df.loc['RHip','X'],df.loc['RKnee','X'])
    mass_elements.loc['R_upperLeg','Y'] = gravityMean2(df.loc['RHip','Y'],df.loc['RKnee','Y'])
    mass_elements.loc['R_lowerLeg','X'] = gravityMean2(df.loc['RKnee','X'],df.loc['RAnkle','X'])
    mass_elements.loc['R_lowerLeg','Y'] = gravityMean2(df.loc['RKnee','Y'],df.loc['RAnkle','Y'])
    mass_elements.loc['L_upperLeg','X'] = gravityMean2(df.loc['LHip','X'],df.loc['LKnee','X'])
    mass_elements.loc['L_upperLeg','Y'] = gravityMean2(df.loc['LHip','Y'],df.loc['LKnee','Y'])
    mass_elements.loc['L_lowerLeg','X'] = gravityMean2(df.loc['LKnee','X'],df.loc['LAnkle','X'])
    mass_elements.loc['L_lowerLeg','Y'] = gravityMean2(df.loc['LKnee','Y'],df.loc['LAnkle','Y'])
    mass_elements.loc['R_foot','X'] = df.loc['RAnkle','X']
    mass_elements.loc['R_foot','Y'] = df.loc['RAnkle','Y']
    mass_elements.loc['L_foot','X'] = df.loc['LAnkle','X']
    mass_elements.loc['L_foot','Y'] = df.loc['LAnkle','Y']
    #
    mass_elements.loc[:,'mass'] = [8,46,4,3,1,7,6,2,4,3,1,7,6,2]
    center_of_massX = np.dot(mass_elements['X'],mass_elements['mass'])/sum(mass_elements.loc[mass_elements['X']!=0,'mass'])
    center_of_massY = np.dot(mass_elements['Y'],mass_elements['mass'])/sum(mass_elements.loc[mass_elements['Y']!=0,'mass'])
    #img = cv2.circle(base_img, (int(center_of_massX), int(center_of_massY)), 7, (255, 255, 255), thickness=5, lineType=cv2.LINE_8, shift=0)
    #for parts in mass_ind:
    #    img = cv2.circle(img, (int(mass_elements.loc[parts,'X']), int(mass_elements.loc[parts,'Y'])), 7, (255, 255, 120), thickness=1, lineType=cv2.LINE_8, shift=0)
    return center_of_massX,center_of_massY


##################################################################################################    
start_time = time.time()
mass_ind = ['head','body',\
            'R_upperArm','R_lowerArm','R_parm','R_upperLeg','R_lowerLeg','R_foot',\
            'L_upperArm','L_lowerArm','L_parm','L_upperLeg','L_lowerLeg','L_foot']
mass_elements = pd.DataFrame(index=mass_ind)

res_df_X = pd.DataFrame();
res_df_Y = pd.DataFrame();

# ディレクトリの下のファイル名のリストを取得する。ー＞　リスト形式
# 変数の型はtype(変数)で見る。type(df)など。
path = workdir+json_file_dir
DLfiles = os.listdir(path)
print(type(DLfiles)) # <class 'list'>
print(DLfiles) # ['dir1', 'dir2', 'file1', 'file2.txt', 'file3.jpg']
DLfiles2 = [f for f in DLfiles if os.path.isfile(os.path.join(path, f))] #dirを除く
print(DLfiles2)   # ['file1', 'file2.txt', 'file3.jpg']
# jsonxファイルのみのリストを作る。システムファイルを除去。
DLfiles3 = []
for file in DLfiles2:
    base, ext = os.path.splitext(file) #拡張子を分離ー＞これは使える！
    if ext == '.json':
#        print('file:{},ext:{}'.format(file,ext))
        DLfiles3.append(file)
DLfiles4 = sorted(DLfiles3)
print('file:{},ext:{}'.format(file,ext))
        
for json_file in DLfiles4:
    #image_file =  '00540.jpg'
    json_prefix = json_file.replace('.json','') 
    # jsonのロード
    with open(workdir+json_file_dir+json_file, 'r') as f:
        data = json.load(f)
    #json->dataFrameへ変換
    if data['people']!=[]:
        data1 = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1,3)
        '''
        df = pd.DataFrame(data1, columns=['X','Y','P'], index=["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "MidHip", "RHip", \
           "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEr", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"])
        '''
        df = pd.DataFrame(data1, columns=['X','Y','P'], index=["Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist", "RHip", \
           "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye",  "LEye", "REar", "LEar"])
            
        CoGX,CoGY = calc_CoG(df)
        df = pd.concat([df,pd.DataFrame({'X':CoGX,'Y':CoGY,'P':0},index=['CenterOfGravity'])])  

        df.columns = [json_prefix+'_X',json_prefix+'_Y',json_prefix+'_P']
        res_df_X = pd.concat([res_df_X,df[json_prefix+'_X']],axis=1)
        res_df_Y = pd.concat([res_df_Y,df[json_prefix+'_Y']],axis=1)     
        
    else:
        print(json_file+' : '+'json file is empty....... skipped.')







########################### 計算 #############################################################

import math
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import japanize_matplotlib



#腰の角度計算
def west_angle(SX,SY,HX,HY,AX,AY):
  tan1 = (HY-SY)/(HX-SX)
  tan2 = (AY-HY)/(AX-HX)

  atan1 = abs(np.arctan(tan1)*180/math.pi)
  atan2 = abs(np.arctan(tan2)*180/math.pi)

  result = atan1 + atan2

  return result

#回帰直線の傾き計算
def Slope(x,y):
    n=len(x)
    t_xy=sum(x*y)-(1/n)*sum(x)*sum(y)
    t_xx=sum(x**2)-(1/n)*sum(x)**2
    slope=t_xy/t_xx
    
    return slope

#腰の点数化
def West_score(angle):
    if(angle <= 120):
        return 5
    elif(angle <= 130):
        return 4
    elif(angle <= 140):
        return 3
    elif(angle <= 150):
        return 2
    elif(angle <= 135):
        return 1
    else:
        return 0
    
    
#左手の点数化    
def LWrist_score(slope):
    if(slope > 5):
        return 5
    elif(slope > 0):
        return 4
    elif(slope > -5):
        return 3
    elif(slope > -10):
        return 2
    elif(slope > -15):
        return 1
    else:
        return 0

#肩の点数化
def Shoulder_score(slope):
    if(slope > 0):
        return 5
    elif(slope > -5):
        return 4
    elif(slope > -10):
        return 3
    elif(slope > -15):
        return 2
    elif(slope > -20):
        return 1
    else:
        return 0

#右足の得点化
def RAnkle_score(slope):
    if(abs(slope) <= 15):
        return 5
    elif(abs(slope) <=30):
        return 4
    elif(abs(slope) <=45):
        return 3
    elif(abs(slope) <=60):
        return 2
    elif(abs(slope) <=75):
        return 1
    else:
        return 0

#右ケツと重心の得点化
def RHip_COG_score(sa):
    if(sa <= -5):
        return 5
    elif(sa <= 0):
        return 4
    elif(sa <= 5):
        return 3
    elif(sa <= 10):
        return 2
    elif(sa <= 15):
        return 1
    else:
        return 0
    
Score_list = []
#####座標を読み込む#####
df_x =res_df_X
df_y =res_df_Y
#腰の角度
##########################
index_X = "00045_X"#蹴る瞬間
df2 = df_x.filter(items=['RShoulder','RHip','RAnkle'],axis=0)
df3 = df2[index_X]
df_RShoulder_X = df3["RShoulder"]
df_RHip_X = df3["RHip"]
df_RAnkle_X = df3["RAnkle"]
#print(df_RShoulder_X,df_RHip_X,df_RAnkle_X)

index_Y =index_X.replace('X','Y')
#print(string_new)

df4 = df_y.filter(items=['RShoulder','RHip','RAnkle'],axis=0)
df5 = df4[index_Y]
df_RShoulder_Y = df5["RShoulder"]
df_RHip_Y = df5["RHip"]
df_RAnkle_Y = df5["RAnkle"]

angle = west_angle(df_RShoulder_X,df_RShoulder_Y,df_RHip_X,df_RHip_Y,df_RAnkle_X,df_RAnkle_Y)
#print(angle)

#逆手の高さ傾き
###########################
##############軸足踏み込んでからボールに足が当たるまで#################################################################################################################
slope_index = ["00031_Y","00032_Y","00033_Y","00034_Y","00035_Y","00036_Y","00037_Y","00038_Y","00039_Y","00040_Y","00041_Y","00042_Y","00043_Y","00044_Y","00045_Y"]#
#slope_index = "00019_Y"
slope_idx = []
for i in slope_index:
    slope_idx.append(int(i.replace("_Y", "")))    
slope_idx = np.array(slope_idx)

df_Lwrist = df_y.filter(items=["LWrist"],axis=0)
df_Lwrist = df_Lwrist[slope_index]
list_Lwrist = df_Lwrist.to_numpy().tolist()
list_Lwrist = list_Lwrist[0]
np_Lwrist = np.array(list_Lwrist)
print(np_Lwrist)
print(slope_idx)
slope_Lwrist = Slope(slope_idx,np_Lwrist)

print(slope_Lwrist)

#肩の高さ傾き
###########################

df_RShoulder = df_y.filter(items=["RShoulder"],axis=0)
df_LShoulder = df_y.filter(items=["LShoulder"],axis=0)
df_RShoulder = df_RShoulder[slope_index]
df_LShoulder = df_LShoulder[slope_index]
np_RShoulder = np.array(df_RShoulder)[0]
np_LShoulder = np.array(df_LShoulder)[0]
np_Shoulder =[]
for i in range(0,len(np_RShoulder)):
    np_Shoulder.append(max(np_RShoulder[i],np_LShoulder[i])-min(np_RShoulder[i],np_LShoulder[i]))

slope_Shoulder = Slope(slope_idx,np_Shoulder)


#########当たってから振り切るまで###########################################################################################
RAnkle_index = ["00045_Y","00046_Y","00047_Y","00048_Y","00049_Y","00050_Y","00051_Y","00052_Y","00053_Y","00054_Y","00055_Y","00056_Y","00057_Y"]
RAnkle_idx = []
for i in RAnkle_index:
    RAnkle_idx.append(int(i.replace("_Y", "")))

RAnkle_idx = np.array(RAnkle_idx)

RAnkle = df_y.filter(items=["RAnkle"],axis=0)
RAnkle = RAnkle[RAnkle_index]#指定した番号のデータの抽出
np_RAnkle = RAnkle.to_numpy()[0]
slope_RAnkle = Slope(RAnkle_idx,np_RAnkle)
#print(RAnkle_score(slope_RAnkle))
#print(np_RAnkle)



RHip_index = ["00045_X"]############蹴る瞬間##############################################################
RHip_idx = []
for i in RHip_index:
    RHip_idx.append(int(i.replace("_X","")))
    
RHip_idx = np.array(RHip_idx)

RHip = df_x.filter(items=["RHip"],axis=0)
RHip = RHip[RHip_index]
RHip = RHip.to_numpy().tolist()[0]
#print(RHip)



COG_index = ["00045_X"]#####蹴る瞬間#########################################################################
COG_idx = []
for i in COG_index:
    COG_idx.append(int(i.replace("_X","")))
    
COG_idx = np.array(COG_idx)

COG = df_x.filter(items=["CenterOfGravity"],axis=0)
COG = COG[COG_index]
COG = COG.to_numpy().tolist()[0]

#print(COG)
RHip_minus_COG = RHip[0] - COG[0]
#print(RHip_COG_score(RHip_minus_COG))


# 多角形を閉じるためにデータの最後に最初の値を追加する。
values = [West_score(angle),Shoulder_score(slope_Shoulder),LWrist_score(slope_Lwrist),RAnkle_score(slope_RAnkle),RHip_COG_score(RHip_minus_COG)]
#print(Score_list)
labels = ["腰の角度","肩の傾き","逆手の傾き","蹴る足の高さ","重心"]
radar_values = np.concatenate([values, [values[0]]])
# プロットする角度を生成する。
angles = np.linspace(0, 2 * np.pi, len(labels) + 1, endpoint=True)
# メモリ軸の生成
rgrids = [0, 1, 2, 3, 4, 5]


fig = plt.figure(facecolor="w")
# 極座標でaxを作成
ax = fig.add_subplot(1, 1, 1, polar=True)
# レーダーチャートの線を引く
ax.plot(angles, radar_values)
#　レーダーチャートの内側を塗りつぶす
ax.fill(angles, radar_values, alpha=0.2)
# 項目ラベルの表示
ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
# 円形の目盛線を消す
ax.set_rgrids([])
# 一番外側の円を消す
ax.spines['polar'].set_visible(False)
# 始点を上(北)に変更
ax.set_theta_zero_location("N")
# 時計回りに変更(デフォルトの逆回り)
ax.set_theta_direction(-1)

# 多角形の目盛線を引く
for grid_value in rgrids:
    grid_values = [grid_value] * (len(labels)+1)
    ax.plot(angles, grid_values, color="gray",  linewidth=0.5)

# メモリの値を表示する
for t in rgrids:
    # xが偏角、yが絶対値でテキストの表示場所が指定される
    ax.text(x=0, y=t, s=t)

# rの範囲を指定
ax.set_rlim([min(rgrids), max(rgrids)])

######名前変更####################################################################################
ax.set_title("成功例", pad=20)
plt.show()


end_time = time.time()
#print(end_time-start_time)




















