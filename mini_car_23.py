import keyboard
import os
import sys
import time
import struct
import socket
import threading
import cv2
from IPy import IP
from websocket import create_connection
import netifaces
import json
import numpy as np
import math
import matplotlib.pyplot as plt


def GetIpList():
    broadcast_list = []
    routingNicName = netifaces.gateways()['default'][netifaces.AF_INET][1]
    for interface in netifaces.interfaces():
        try:
            routingIPAddr = netifaces.ifaddresses(
                interface)[netifaces.AF_INET][0]['addr']
            # TODO(Guodong Ding) Note: On Windows, netmask maybe give a wrong result in 'netifaces' module.
            routingIPNetmask = netifaces.ifaddresses(
                interface)[netifaces.AF_INET][0]['netmask']
            # print('IP：',routingIPAddr)
            # print('mask：', routingIPNetmask)
            broadcast = IP(routingIPAddr).make_net(
                routingIPNetmask).broadcast().strNormal()
            # print('broadcast:',broadcast)
            broadcast_list.append(broadcast)
        except KeyError:
            pass
    return broadcast_list


def MiniCarScan(scan_num=3,id = None):
    udp_socekt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socekt.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, True)
    msg = b"###ss-discover#SenseRover_mini_v1.0###"
    broadcast_list = GetIpList()
    print("IP_LIST:",broadcast_list)
    car_dict = {}
    for ip in broadcast_list:
        udp_socekt.settimeout(1)
        for i in range(scan_num):
            udp_socekt.sendto(msg, (ip, 21567))
            while True:
                try:
                    recv, addr = udp_socekt.recvfrom(1024)
                    recv = recv.decode()
                    car_msg = recv.split('###')[1].split("#")
                    if car_msg[5] == '':
                        car_dict[car_msg[4]] = addr[0]
                        if id != None and car_msg[4] == id:
                            return car_dict
                except:
                    break
    return car_dict


class MiniCar:
    __PING_CODE = 0x01
    __PONG_CODE = 0x02
    __REMOTE_CODE = 0x03
    __TEXT_CODE = 0x04
    __TEXT_RESPON_CODE = 0x05
    __SENSORS_CONTROL_CODE = 0x06
    __SENSORS_UPLOAD_CODE = 0x07

    def __init__(self, id=None, ip=None):
        self.__ws = None
        self.__ping_event = threading.Event()
        self.__ping_event.set()
        self.__ping_thread = None
        self.__recv_event = threading.Event()
        self.__recv_event.set()
        self.__recv_thread = None
        self.exec_flag = False
        self.sensors_data = {"RGBL": (0, 0, 0, 0), "DISTANCE": (0, 0), "INTENSITY": (0, 0, 0, 0, 0), "VOL": 0, "ANGLE": 0, "LR_SPEED": (0, 0), "MILESTONE": 0,
                    "RGB_LIGHT": (0, 0, 0), "MOTOR_SPEED": (0, 0), "MOTOR_TIME": 0, "STATUS": 0, "LR_ENCODER": (0, 0), "TIME_STAMP": 0}
        if id is not None:
            self.connectWithID(id)
        elif ip is not None:
            self.connectWithIP(ip)

    def __del__(self):
        if not self.__ping_event.isSet():
            self.__ping_event.set()
            self.__ping_thread.join()
        if not self.__recv_event.isSet():
            self.__recv_event.set()
            self.__recv_thread.join()
        if self.__ws is not None:
            self.__ws.close()

    def connectWithID(self, id):
        car_dict = MiniCarScan(id = id)
        print(car_dict)
        if id in car_dict:
            self.connectWithIP(car_dict[id])
        else:
            print("can't find %s" % (id))

    def connectWithIP(self, ip):
        self.__del__()
        self.__ws = create_connection("ws://" + ip + ":8266")
        self.__ws.settimeout = 1
        self.__ping_thread = threading.Thread(target=self.__ping, args=(15,))

        self.__ping_thread.daemon =True
        self.__ping_thread.start()
        self.__recv_thread = threading.Thread(target=self.__recive)
        self.__recv_thread.daemon =True
        self.__recv_thread.start()

    def __ping(self, ping_time):
        self.__ping_event.clear()
        while not self.__ping_event.isSet():
            self.__ws.send(b'\x01\x00')
            self.__ping_event.wait(ping_time)
        print("stop ping thread")

    def __recive(self):
        self.__recv_event.clear()
        while not self.__recv_event.isSet():
            s = self.__ws.recv()
            if len(s) < 1:
                continue
            # s = str(s,encoding='utf-8')
            if s[0] == self.__PONG_CODE:
                print('get pong')
            elif s[0] == self.__TEXT_RESPON_CODE:
                print(str(s[2:], encoding='utf-8'))
                if s[2] == ord('S'):
                    self.exec_flag = True
                elif s[2] == ord('E'):
                    self.exec_flag = False
            elif s[0] == self.__SENSORS_UPLOAD_CODE:
                self.sensors_data.update(
                    RGBL=(struct.unpack('<H', s[2:4])[0], struct.unpack('<H', s[4:6])[
                          0], struct.unpack('<H', s[6:8])[0], struct.unpack('<H', s[8:10])[0]),
                    DISTANCE=(struct.unpack('<H', s[10:12])[
                              0], struct.unpack('<H', s[12:14])[0]),
                    INTENSITY=(struct.unpack('<H', s[14:16])[0], struct.unpack('<H', s[16:18])[0], struct.unpack('<H', s[18:20])[0],
                               struct.unpack('<H', s[20:22])[0], struct.unpack('<H', s[22:24])[0]),
                    VOL=struct.unpack('<H', s[24:26])[0],
                    ANGLE=struct.unpack('<b', s[26:27])[0],
                    LR_SPEED=(struct.unpack('<h', s[27:29])[
                              0], struct.unpack('<h', s[29:31])[0]),
                    MILESTONE=struct.unpack('<Q', s[31:39])[0],
                    RGB_LIGHT=(struct.unpack('<B', s[39:40])[0], struct.unpack(
                        '<B', s[40:41])[0], struct.unpack('<B', s[41:42])[0]),
                    MOTOR_SPEED=(struct.unpack('<b', s[42:43])[
                                 0], struct.unpack('<b', s[43:44])[0]),
                    MOTOR_TIME=struct.unpack('<I', s[44:48])[0],
                    STATUS=struct.unpack('<B', s[48:49])[0],
                    LR_ENCODER=(struct.unpack('<q', s[49:57])[
                                0], struct.unpack('<q', s[57:65])[0]),
                    TIME_STAMP=struct.unpack('<I', s[65:])[0]
                )
        print("stop recive thread")

    def car_remote(self, l_speed=127, r_speed=127, go_time=1, angle=127, block=False):
        def int2bytes(val):
            return val.to_bytes(1, byteorder='little', signed=True)
        if go_time is None:
            self.__ws.send(bytes([self.__REMOTE_CODE, 0]) + int2bytes(
                l_speed) + int2bytes(r_speed)+int2bytes(angle), opcode=0x02)
        else:
            ms = bytes([self.__REMOTE_CODE, 0]) + int2bytes(l_speed) + int2bytes(r_speed) + \
                int2bytes(angle)+(int(go_time*1000)).to_bytes(4,
                                                              byteorder='little', signed=False)
            # print(ms)
            self.__ws.send(ms, opcode=0x02)
            if block:
                time.sleep(go_time)

    def set_senseors_upload_time(self, upload_time=0.5):
        if upload_time <= 0:
            self.__ws.send(bytes([self.__SENSORS_CONTROL_CODE, 0]) + b'E')
        else:
            self.__ws.send(bytes([self.__SENSORS_CONTROL_CODE, 0]) + b'S' + (
                int(upload_time*1000)).to_bytes(4, byteorder='little', signed=False))

    def go(self, l_speed, r_speed, go_time=None, block=False):
        self.car_remote(l_speed=l_speed, r_speed=r_speed,
                        go_time=go_time, block=block)

    def get_time_stamp(self):
        return (self.sensors_data['TIME_STAMP'])

    def turn(self, angle):
        self.car_remote(angle=angle)

    def get_encoders(self):
        return (self.sensors_data['LR_ENCODER'])

    def get_speed(self):
        return (self.sensors_data['LR_SPEED'])

    def get_distance(self):
        return (self.sensors_data['DISTANCE'])

    def exec(self, code):
        if not isinstance(code,bytes):
            return
        self.exec_flag = True
        self.__ws.send(bytes([self.__TEXT_CODE, 0])+ b'\x04'+code)

    def disconnect(self):
        self.__del__()


class carMove:  # Localization
    # mini参数设置

    # dt越大越接近视觉测量值
    # mini定位
    def __init__(self, Xp_car0, Xp_vector0, cam, lr, lf, K, H1=None, H2=None):

        self.lr = lr  # cm
        self.lf = lf  # cm
        self.K = K  # 互补滤波系数,越大则越接近视觉的测量值,
        # 单应性矩阵初始化
        if H1 is not None:
            self.H_cam1 = H1
        else:
            self.H_cam1 = np.loadtxt('Homography1.txt')
        if H2 is not None:
            self.H_cam2 = H2
        else:
            self.H_cam2 = np.loadtxt('Homography2.txt')
        # 计算初始位置
        self.Xw = self.getVisionPosition(Xp_car0, Xp_vector0, cam)

    def __findPosition(self, H, Xpi):
        # 单应性变换
        Xp_ = np.array([Xpi[0], Xpi[1], 1])
        Xw_ = np.dot(H, Xp_)
        Xw = Xw_ / Xw_[2]/100

        Xw = np.array([Xw[0], Xw[1]])
        return Xw

    def getVisionPosition(self, Xp_car, Xp_vector, cam):
        # 从像素坐标计算mini的世界坐标
        if cam == 1:
            H = self.H_cam1
        if cam == 2:
            H = self.H_cam2
        Xp_car = np.array(Xp_car)
        Xp_vector = np.array(Xp_vector)
        Xw_car = (self.__findPosition(H, Xp_car)) * 100  # 米到厘米
        Xw_vector = (self.__findPosition(H, Xp_vector)) * 100  # 米到厘米
        phi = math.atan2(Xw_vector[1] - Xw_car[1], Xw_vector[0] - Xw_car[0])
        self.visionXw = np.array([Xw_car[0], Xw_car[1], phi])
        self.vXw_last = self.visionXw  # 记录这一次的车辆像素坐标
        return self.visionXw

    def angleWeightAverage(self, a, b, ka, kb):
        if np.abs(a - b) < np.pi:
            c = ka * a + kb * b
        else:
            c = ka * (a + 2 * np.pi * (a < 0)) + kb * (b + 2 * np.pi * (b < 0))
        if c > np.pi:
            c = c - 2 * np.pi
        return c

    def __filter(self, v, beta, dt):
        # 互补滤波
        dX = np.array([np.cos(self.Xw[2] + beta), \
                       np.sin(self.Xw[2] + beta), \
                       np.sin(beta) / self.lr])
        #        Xw_n0 = self.Xw + dX*v*dt
        #        Xw_n1 = Xw_n0 + self.__K*(self.visionXw - Xw_n0)
        # Xw_n1[0]=(1.0-self.K)*self.Xw[0]+self.K*self.visionXw[0]+dX[0]*v*dt
        # Xw_n1[1]=(1.0-self.K)*self.Xw[1]+self.K*self.visionXw[1]+dX[1]*v*dt
        Xw_n1 = (1.0 - self.K) * self.Xw + self.K * self.visionXw + (1 - self.K) * dX * v * dt
        angle1 = self.Xw[2]
        angle2 = self.visionXw[2]
        angle3 = dX[2] * v * dt

        Xw_n1[2] = self.angleWeightAverage(angle1, angle2, 1 - self.K, self.K) + (1-self.K)*angle3
        if Xw_n1[2] < -np.pi:
            Xw_n1[2] += 2 * np.pi
        elif Xw_n1[2] > np.pi:
            Xw_n1[2] -= 2 * np.pi
        self.Xw = Xw_n1

    def getCarPosition(self, Xp_car, Xp_vector, cam, v, beta, dt, location_mode):
        # 计算mini的世界坐标
        if (Xp_car - np.array([-1, -1])).all() == False or (Xp_car - np.array([0, 0])).all() == False:  # 如果像素坐标没有更新
            self.visionXw = self.Xw  # 则把积分得到的坐标视为其测量值
        #        else:
        #            if self.vXw_last is not None :
        #                if cam == 1:
        #                    H=self.H_cam1
        #                if cam == 2:
        #                    H=self.H_cam2
        #                Xp_car=np.array(Xp_car)
        #                xwc=self.__findPosition(H,Xp_car)*100
        #                xwclst=np.array([self.vXw_last[0],self.vXw_last[1]])
        #                if np.linalg.norm(xwclst-xwc)>3:
        #                    self.visionXw=self.Xw
        #                else:
        #                    self.getVisionPosition(Xp_car,Xp_vector,cam)

        if location_mode == 0:
            self.visionXw = self.Xw
            print('location_mode',location_mode)
        self.__filter(v, beta, dt)
        return self.Xw

    def carError(self, destinatePoint, nowPoint):
        verticalError = np.linalg.norm(destinatePoint - nowPoint[0:2])
        destinateAngle = np.arctan2(destinatePoint[1] - nowPoint[1], destinatePoint[0] - nowPoint[0])
        horizontalError = destinateAngle - nowPoint[2]
        if horizontalError > np.pi:
            horizontalError -= 2 * np.pi
        elif horizontalError < -np.pi:
            horizontalError += 2 * np.pi
        horizontalError = horizontalError / np.pi * 180
        return (verticalError, horizontalError)



# cameraNum = 0  # 相机序号
# planner = trajectoryPlanning()  # 轨迹规划
# lr = 6  # 4.2 #cm#转弯中心到后轮的距离
# lf = 4  # 5.3 #cm#转弯中心到前轮的距离
# K_filter = 0.4  # 0-1,越大越相信相机
# # !!!!!!!!!!INIT!!!!!!!!!!!!!!!!!!
# # refer_init(cameraNum)

# refer_frame = cv2.imread("blank.jpg")
# cv2.imshow('blank', refer_frame)
# cv2.waitKey(1000)
# cv2.destroyAllWindows()
# red_car = RedCar(refer_frame)  # 检测小车的对象
# # blue_car = BlueCar(refer_frame)
# cap = cv2.VideoCapture(cameraNum, cv2.CAP_DSHOW)
# _ = cap.set(3, 1920)
# _ = cap.set(4, 1080)
# _ = cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))

# carPointList = np.zeros(shape=(50, 2))
# headPointList = np.zeros(shape=(50, 2))

# carInitialPoint = np.array([0, 0])
# carInitialHead = np.array([0, 0])
# i = 0
# ######################寻找初始点
# while i < 50:
#     ret, img_current = cap.read()
#     # img_current = cv2.resize(img_current,(int(1920/2),int(1080/2)))
#     # cv2.imshow("frame",img_current)
#     # print("come")
#     red_car.car_info(img_current)
#     if red_car.car_point[0] > 0 and red_car.car_point[1] > 0:
#         carPointList[i, :] = np.array(red_car.car_point)
#         headPointList[i, :] = np.array(red_car.head_point)
#         i += 1
# carInitialPoint1 = [np.median(carPointList[:, 0]), np.median(carPointList[:, 1])]
# carInitialHead1 = [np.median(headPointList[:, 0]), np.median(headPointList[:, 1])]
# i = 0
# j = 0
# a = time.time()
# while i < 50:
#     j += 1
#     ret, img_current = cap.read()

#     red_car.car_info(img_current)
#     cv2.circle(img_current, (int(red_car.car_point[0]), int(red_car.car_point[1])), 6, (213, 213, 0), -1)
#     cv2.circle(img_current, (int(red_car.head_point[0]), int(red_car.head_point[1])), 6, (0, 0, 213), -1)
#     img_resize = cv2.resize(img_current, (640, 360))
#     cv2.imshow('blank', img_resize)
#     cv2.waitKey(30)
#     if np.linalg.norm(np.array(red_car.car_point) - carInitialPoint1) < 10:
#         carInitialPoint = carInitialPoint + np.array(red_car.car_point)
#         carInitialHead = carInitialHead + np.array(red_car.head_point)
#         i += 1
# a = time.time() - a
# print('fps', j / a)
# if i / j < 0.8:
#     print('Vision system need calibrate')
# print('Vision quality is', i / j)
# cv2.destroyAllWindows()
# carInitialPoint = carInitialPoint / 50
# carInitialHead = carInitialHead / 50

# move = carMove(carInitialPoint, carInitialHead, 1, lr, lf, K_filter)  # 视觉融合定位
# xw = move.getCarPosition(red_car.car_point, red_car.head_point, 1, 0, 0, 0.05, 1)  # x坐标in world
# vision = move.getVisionPosition(red_car.car_point, red_car.head_point, 1)  # 纯视觉的x坐标 in world
# print('constructing communication with car...')
# car = MiniCar(id='0000S2')
# # move.Xw=np.array([0.,0.,0.])
# # v=0
# # beta=0
# # dt=0.05
# # xw=move.getCarPosition([0,0], [0,0], 1, v, beta, dt)

# speed = 0
# ang = 0
# update = 0
# print('start')
# car.set_senseors_upload_time(0.05)
# # while True:
# #     time.sleep(0.01)
# #     if car.get_speed()[0] != 0 or car.get_speed()[1] != 0:
# #         print('[%d]speed:' % ((time.time() * 1000 % 100000)), car.get_speed())
# #         print('[%d]encoders:' %
# #               ((time.time()*1000 % 100000)), car.get_encoders())

# # car.disconnect()

# lasttime = 0
# encoderS = car.get_encoders()
# encoderS = car.get_encoders()
# encoderLold = encoderS[0]
# encoderRold = encoderS[1]
# wheelDiameter = 4  # cm
# x = 0
# y = 0
# v = 0
# dt = 0.05
# phi = 0
# xhis = x
# yhis = y
# carTimeOld = car.get_time_stamp()
# time.sleep(0.1)
# carTimeOld = car.get_time_stamp()
# xhis = [0]
# yhis = [0]
# dthis = [0]
# xvhis = [0]
# yvhis = [0]
# anghis = [0]
# phiMhis = [0]
# phiVhis = [0]
# destinatePointXhis = [0]
# destinatePointYhis = [0]
# visionCarPointHis = [0, 0]
# vhis = [0]
# err = 0

# stopflag = 0

# num = 0
# posController = PIDcontroller(1.5, 0, 0)
# angController = PIDcontroller(1.5, 0, 0.02)

# while stopflag == 0:
#     num += 1
#     if time.time() - lasttime > 0.050:
#         lasttime = time.time()
#         ret, img_current = cap.read()
#         # img_current = cv2.resize(img_current,(int(1920/2),int(1080/2)))
#         # cv2.imshow("frame",img_current)
#         # print("come")
#         red_car.car_info(img_current)
#         # blue_car.car_info(img_current)
#         try:
#             carTime = car.get_time_stamp()
#             dt = (carTime - carTimeOld) / 1000
#             print('dt:', dt)
#             carTimeOld = carTime
#             encoderS = car.get_encoders()
#             encoderL = encoderS[0]
#             encoderR = encoderS[1]
#             if dt == 0:
#                 continue
#             else:
#                 v = ((encoderL - encoderLold) + (encoderR - encoderRold)) / 2 / 840 * 3.14 * 4 / dt
#                 if v > 300:
#                     v = 0
#             if dt > 1:
#                 dt = 0
#             encoderLold = encoderL
#             encoderRold = encoderR
#             sigma = ang / 180 * 3.14  # 舵机角度
#             beta = np.arctan(lr / (lf + lr) * np.tan(sigma)) / 1.2  # 前进方向与车头方向夹角
#             # x = x + v*np.cos(phi+beta)*dt
#             # y = y + v*np.sin(phi+beta)*dt
#             # phi = phi + v/lr*np.sin(beta)*dt
#             vision = move.getVisionPosition(red_car.car_point, red_car.head_point, 1)
#             xv = vision[0]
#             yv = vision[1]
#             if np.linalg.norm([xhis[-1] - xv, yhis[-1] - yv]) > 5 and num > 100:
#                 location_mode = 0  # 纯里程计定位
#             else:
#                 location_mode = 1  # 融合定位

#             xw = move.getCarPosition(red_car.car_point, red_car.head_point, 1, v, beta, dt,
#                                      location_mode)  # 3维向量，x,y,phi
#             phiVision = math.atan2(red_car.head_point[1] - red_car.car_point[1],
#                                    red_car.head_point[0] - red_car.car_point[0])
#             # xw = [xv,yv,phiVision]
#             dthis.append(dt)

#             visionCarPointHis.append(red_car.car_point)

#             x = xw[0]
#             y = xw[1]
#             xw = np.array(xw)
#             destinatePoint = planner.planning()
#             verticalError, horizontalError = move.carError(destinatePoint, xw)
#             speed = posController.controller(verticalError)
#             ang = angController.controller(horizontalError)
#             if verticalError < 5:
#                 ang = 0
#                 speed = 0
#             # speed,ang = move.carController(destinatePoint,xw)
#             # if keyboard.is_pressed('w'):
#             #     speed += 1
#             # elif keyboard.is_pressed('s'):
#             #     speed -= 1
#             # if keyboard.is_pressed('a'):
#             #     ang -= 5
#             # elif keyboard.is_pressed('d'):
#             #     ang += 5
#             # else:
#             #     ang = ang-int(np.sign(ang)*5)
#             #     if abs(ang) <5:
#             #         ang = 0
#             if speed > 30:
#                 speed = 30
#             elif speed < -100:
#                 speed = -100
#             if speed < 10 and speed > 0:
#                 speed = 10
#             if speed > -10 and speed < 0:
#                 speed = -10
#             if ang > 35:
#                 ang = 35
#             elif ang < -35:
#                 ang = -35
#             if keyboard.is_pressed('f'):
#                 ang = 0
#                 speed = 0
#             if keyboard.is_pressed('p'):
#                 stopflag = 100
#             car.car_remote(l_speed=int(speed), r_speed=int(speed),
#                            go_time=1, angle=int(ang), block=False)
#             print('SPEED', speed)

#             phiMix = xw[2]
#             xhis.append(x)
#             yhis.append(y)
#             vhis.append(v)
#             xvhis.append(xv)
#             yvhis.append(yv)
#             anghis.append(ang / 180 * 3.14 / 1.5)
#             phiMhis.append(phiMix)
#             phiVhis.append(phiVision)
#             destinatePointXhis.append(destinatePoint[0])
#             destinatePointYhis.append(destinatePoint[1])




#         except Exception as e:
#             print(e)
#             car.car_remote(l_speed=0, r_speed=0, go_time=0, angle=0, block=False)
#             break
# car.disconnect()
# print("&&&&&&&&&&&&&&&&&&&&&&", err)
# print("&&&&&&&&&&&&&&&&&&&&&&", len(xhis) / 1.0)

# print("&&&&&&&&&&&&&&&&&&&&&&", err * 1.0 / len(xhis) / 1.0)
# for i in range(len(xhis)):
#     print(xhis[i], yhis[i])

# plt.figure(1, figsize=(6, 6))
# # plt.plot(xhis,yhis)
# plt.scatter(xhis, yhis, alpha=0.2)
# plt.show()

# plt.figure(2, figsize=(6, 6))
# # plt.plot(xhis,yhis)
# plt.scatter(xvhis, yvhis, alpha=0.2)
# plt.show()

# # sudo / home/lhs/Software/anaconda3/bin/python / \
# #     mnt/Personal/SenseTime/lotus_top/mini_car.py
