import glob
import cv2
from time import sleep
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import time
from PIL import Image
import numpy as np
import CLI

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


def convert_rgb_to_iyu2(acts):
    c, h, w = acts.shape
    iyu2 = torch.zeros_like(acts)
    assert c == 3
    r, g, b = acts[0], acts[1], acts[2]
    y = 0.213 * r + 0.715 * g + 0.072 * b
    u = -0.117 * r - 0.394 * g + 0.511 * b + 128
    v = 0.511 * r - 0.464 * g - 0.047 * b + 128
    iyu2[0], iyu2[1], iyu2[2] =u.round().clamp(0,255), y.round().clamp(0.255), v.round().clamp(0.255)

    return iyu2

def save_bin(file,nary):
    binfile = open(file,'wb')
    for h in range(nary.shape[0]):
        for w in range(nary.shape[1]):
            binfile.write(nary[h][w][0])
            binfile.write(nary[h][w][1])
            binfile.write(nary[h][w][2])
    binfile.close()


def ary2bin(nary):
    dat = b''
    for h in range(nary.shape[0]):
        for w in range(nary.shape[1]):
            dat += nary[h][w][0]
            dat += nary[h][w][1]
            dat += nary[h][w][2]
    return dat


def save_yuv(file,nary):
    binfile = open(file+'yuv','wb')
    for h in range(nary.shape[0]):
        for w in range(nary.shape[1]):
            binfile.write(nary[h][w][1])
            binfile.write(nary[h][w][0])
            binfile.write(nary[h][w][2])
    binfile.close()

def ary2bin_yuv(nary):
    dat = b''
    for h in range(nary.shape[0]):
        for w in range(nary.shape[1]):
            dat += nary[h][w][1]
            dat += nary[h][w][0]
            dat += nary[h][w][2]
    return dat

def getClassList(file):
    with open(file,'r') as f:
        class_str = f.read()
    names = class_str.split("\n")
    return names


def result(socket, img):
    num = CLI.getDataRSP(socket,"93000004",4)
    int_num = int.from_bytes(num, byteorder='little',signed=True)
    if int_num == 0:
        return
    data = CLI.getDataRSP(socket, "93000008", 4*6*int_num)
    int_data = []

    for i in range(int_num*6):
        tmp = data[i*4:(i*4)+4]
        int_tmp = int.from_bytes(tmp, byteorder='little',signed=True)
        int_data.append(int_tmp)

    for i in range(int_num):
        print("x_min "+ str(int_data[(i*6)])+", "
            +"x_max " + str(int_data[(i*6)+1])+", "
            +"y_min " + str(int_data[(i*6)+2])+", "
            +"y_max " + str(int_data[(i*6)+3])+", "
            +"score " + str(int_data[(i*6)+4])+", "
            +"class " + str(int_data[(i*6)+5]))
        x_min = int((int_data[(i*6)] * img.shape[1] ) /256)
        x_max = int((int_data[(i*6)+1] * img.shape[1]  ) /256)
        y_min = int((int_data[(i*6)+2] * img.shape[0]  ) /256)
        y_max = int((int_data[(i*6)+3] * img.shape[0]  ) /256)
        score = str(round(int_data[(i*6)+4]*100 /256 , 1))
        cv2.rectangle(img, (x_min,y_min), (x_max,y_max), (0, 0, 255), 1)
        position = (x_min, y_min)

        class_name = getClassList("voc-model-labels.txt")
        cv2.putText(img,class_name[int_data[(i*6)+5]] + score ,position ,cv2.FONT_HERSHEY_PLAIN,1,(0,0,255,128),1)
        CLI.setDataRSP(socket,"93000004",4,b'\x00\x00\x00\x00')

images = glob.glob('./img/*.jpg')
print(images)

idx=0

for imgname in images:
    img_bgr = cv2.imread(imgname, cv2.IMREAD_COLOR)

    # USE my YUV converter
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb_300 = cv2.resize(img_rgb, dsize=(304, 300), interpolation=cv2.INTER_LINEAR)
    img = np.transpose(img_rgb_300, (2, 0, 1))
    tensor_img = torch.from_numpy(img)
    img = convert_rgb_to_iyu2(tensor_img)
    npimg = img.numpy()
    new_npimg = np.transpose(npimg, (1, 2, 0))
    fname = imgname + ".bin"
    save_bin(fname, new_npimg)
    new_data = ary2bin(new_npimg)

    # USE cv COLOR_BGR2YUV converter
    # img_bgr_300 = cv2.resize(img_bgr, dsize=(304, 300), interpolation=cv2.INTER_LINEAR)
    # img_yuv_300 = cv2.cvtColor(img_bgr_300, cv2.COLOR_BGR2YUV)
    # new_data_yuv = ary2bin_yuv(img_yuv_300)
    # fname_yuv = imgname + "py.bin"
    # save_yuv(fname_yuv, img_yuv_300)



    socket = CLI.NetCon('localhost', 5557)
    CLI.setDataRSP(socket, "90000000", 304*300*3, new_data)
    flag = b'\x01\x00\x00\x00'
    CLI.setDataRSP(socket, "93000000", 4, flag);

    while True:
        flag = CLI.getDataRSP(socket, "93000000", 4)
        if flag != b'\x01\x00\x00\x00':
            break
        time.sleep(0.01)

    result(socket, img_bgr)
    print(imgname)
    cv2.imshow("VideoFrame", img_bgr)
    cv2.waitKey(0)

cv2.waitKey(1000)
cv2.destroyAllWindows()