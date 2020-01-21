import time
import socket
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from detect_two import BlueCar, RedCar,CyanCar
from detect_two import refer_init
from mini_car_23 import MiniCar, carMove
from trajectoryPlanning import trajectoryPlanning,trajectoryPlanning_fast,trajectoryPlanning_part
from PIDcontroller import PIDcontroller
import threading

class CarControl:
    __park_code = """
turn(35)
go(-20,1.5)
turn(0)
go(-15,1)""".encode()

    __turn_around_code = """
turn(-35)
go(20,1.5)
turn(35)
go(-20,1.3)""".encode()

    def __init__(self,car_num,car_control,move,car,planner,posController,angController):
        self.car_num = car_num
        self.car_control = car_control
        self.move = move
        self.car = car
        self.car_list = []
        self.__my_planner = self.planner = planner
        self.posController = posController
        self.angController = angController
        self.car_location = (0,0,0)
        
        self._carTimeOld = 0
        self._encoderLold = 0
        self._encoderRold = 0
        self._ang = 0
        self._speed = 0
        # 转弯中心到后轮的距离（cm）
        self._lr = 6
        # 转弯中心到前轮的距离（cm）
        self._lf = 4
        self._xhis = [0]
        self._yhis = [0]
        self._dthis = [0]
        self._xvhis = [0]
        self._yvhis = [0]
        self._anghis = [0]
        self._phiMhis = [0]
        self._phiVhis = [0]
        self._destinatePointXhis = [0]
        self._destinatePointYhis = [0]
        self._visionCarPointHis = [0, 0]
        self._vhis = [0]

        self.is_park = False
        self.is_leaving = False
        """
        0:ready to go lap
        1:finish lap
        2:paking
        3:parking finish
        4:leaving
        """
        self.status = 0

    def start(self):
        self.car_control.set_senseors_upload_time(0.05)
        encoderS = self.car_control.get_encoders()
        encoderS = self.car_control.get_encoders()
        self._encoderLold = encoderS[0]
        self._encoderRold = encoderS[1]
        self._carTimeOld = self.car_control.get_time_stamp()
        time.sleep(0.1)
        self._carTimeOld = self.car_control.get_time_stamp()
        
    def run(self):
        # if self.status == 2:
        #     self.__parking()

        carTime = self.car_control.get_time_stamp()
        dt = (carTime - self._carTimeOld) / 1000
        print(self.car_num,'car sensors dt:', dt)
        self._carTimeOld = carTime
        encoderS = self.car_control.get_encoders()
        encoderL = encoderS[0]
        encoderR = encoderS[1]
        if dt == 0:
            return
        else:
            v = ((encoderL - self._encoderLold) + (encoderR - self._encoderRold)) / 2 / 840 * 3.14 * 4 / dt
            if v > 300:
                v = 0
        if dt > 1:
            dt = 0
        self._encoderLold = encoderL
        self._encoderRold = encoderR
        sigma = self._ang * 3.14 / (1.5 * 180)  # 舵机角度，小车转角与实际偏转角为1.5 : 1
        beta = np.arctan(self._lr / (self._lf + self._lr) * np.tan(sigma)) / 1.2  # 前进方向与车头方向夹角
        vision = self.move.getVisionPosition(self.car.car_point, self.car.head_point, 1)
        xv = vision[0]
        yv = vision[1]
        if np.linalg.norm([self._xhis[-1] - xv,self._yhis[-1] - yv]) > 50:
            location_mode = 0  # 纯里程计定位
        else:
            location_mode = 1  # 融合定位
        xw = self.move.getCarPosition(self.car.car_point, self.car.head_point, 1, v, beta, dt,
                                 location_mode)  # 3维向量，x,y,phi
        phiVision = math.atan2(self.car.head_point[1] - self.car.car_point[1],
                               self.car.head_point[0] - self.car.car_point[0])
        # xw = [xv,yv,phiVision]
        self._dthis.append(dt)
        self._visionCarPointHis.append(self.car.car_point)
    
        x = xw[0]
        y = xw[1]
        self.car_location = np.array(xw)
        print(self.car_num,' location:',self.car_location)
        is_end, trajectory = self.planner.planning()
        verticalError, horizontalError = self.move.carError(trajectory, self.car_location)
        self._speed = self.posController.controller(verticalError)
        self._ang = self.angController.controller(horizontalError)
        if verticalError < 5:
            self._ang = 0
            self._speed = 0
            if is_end:
                if self.is_park:
                    self.car_control.exec(self.__park_code)
                    self.is_park = False
                    self.status = 3
                elif self.status == 4 or self.status == 1:
                    self.status = 0

        if self.is_leaving and self.planner.nowPoint == 4:
            self.car_control.exec(self.__turn_around_code)
            self.is_leaving = False

        speed = self._speed
        ang = self._ang

        space = self.__car_min_space()
        if self.status == 1 and space < 40 and 1 < self.planner.nowPoint < 6 and car_status_list[self.car_num] == 1:
            p = space / 50 + 0.2
            speed = self._speed * p
            if self.car_control.get_distance()[1] < 130 or self.car_control.get_distance()[0] < 130:
                p = p * min(self.car_control.get_distance()[0],self.car_control.get_distance()[1]) // 120
                speed = self._speed * p
            self.planner.beginTime += (0.05 * (1 - p))
        print(self.car_num, ' status:', self.status)
        print(self.car_num, ' car_space:', space)
        print(self.car_num,' distance:', self.car_control.get_distance())
        print(self.car_num,' speed,ang:', speed,ang)
        if speed > 100:
            speed = 100
        if ang > 35:
            ang = 35
        elif ang < -35:
            ang = -35
        if not self.car_control.exec_flag and self.status != 3:
            self.car_control.car_remote(l_speed=int(speed), r_speed=int(speed), go_time=1, angle=int(ang), block=False)
        else:
            self.planner.beginTime += 0.05
        self._destinatePointXhis.append(trajectory[0])
        self._destinatePointYhis.append(trajectory[1])
        self._anghis.append(ang / 180 * 3.14 / 1.5)
        self._phiMix = xw[2]
        self._xhis.append(x)
        self._yhis.append(y)
        self._vhis.append(v)
        self._xvhis.append(xv)
        self._yvhis.append(yv)
        self._phiMhis.append(self._phiMix)
        self._phiVhis.append(phiVision)

    def parking(self):
        # self.__parking()
        if self.status == 0:
            self.__parking()
            self.status = 2

    def __parking(self):
        # if self.status == 1:
        #     parking_plan.landmark = np.array([self.__my_planner.landmark[0],[136,225],[35,225],[35,145]])
        #     self.planner = parking_plan
        #     self.planner.start()
        #     self.is_park = True
        # elif self.status == 0:
        parking_plan.landmark = np.array([[136, 225], [35, 215], [35, 145]])
        self.planner = parking_plan
        self.planner.start()
        self.is_park = True

    def leaving(self,planner):
        if self.status == 3:
            self.status = 4
            self.__my_planner = planner
            leaving_plan.landmark = np.array([[35,180],[35,235],[126,235],self.__my_planner.landmark[5],self.__my_planner.landmark[4],self.__my_planner.landmark[0]])
            self.planner = leaving_plan
            self.planner.start()
            self.is_leaving = True

    def lap(self):
        if self.status == 0:
            self.planner = self.__my_planner
            self.planner.start()
            self.status = 1

    def __car_min_space(self):
        min = 999
        for car in self.car_list:
            if np.linalg.norm((self.car_location[0:2]) - (car.car_location[0:2])) < min:
                min = np.linalg.norm((self.car_location[0:2]) - (car.car_location[0:2]))
        return min

# 加载或更新样板图片，通过cameraNum更新
def load_refer_frame(is_update=False, cameraNum=0):
    if is_update:
        refer_init(cameraNum)
    refer_frame = cv2.imread("blank.jpg")
    cv2.imshow('blank',cv2.resize(refer_frame, (640, 360)))
    if is_update:
        cv2.waitKey(10000)
        exit(0)
    else:
        cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return refer_frame


def open_video(cameraNum):
    cap = cv2.VideoCapture(cameraNum, cv2.CAP_DSHOW)
    _ = cap.set(3, 1920)
    _ = cap.set(4, 1080)
    _ = cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS,0)
    return cap


# 寻找初始点
def car_init(cap, car):
    carPointList = np.zeros(shape=(50, 2))
    headPointList = np.zeros(shape=(50, 2))
    carInitialPoint = np.array([0, 0])
    carInitialHead = np.array([0, 0])
    i = 0
    while i < 50:
        ret, img_current = cap.read()
        car.car_info(img_current)
        if car.car_point[0] > 0 and car.car_point[1] > 0:
            carPointList[i, :] = np.array(car.car_point)
            headPointList[i, :] = np.array(car.head_point)
            i += 1
    carInitialPoint1 = [np.median(carPointList[:, 0]), np.median(carPointList[:, 1])]
    i = j = 0
    TimeCount = time.time()
    while i < 50:
        j += 1
        ret, img_current = cap.read()
        car.car_info(img_current)
        cv2.circle(img_current, (int(car.car_point[0]), int(car.car_point[1])), 6, (213, 213, 0), -1)
        cv2.circle(img_current, (int(car.head_point[0]), int(car.head_point[1])), 6, (0, 0, 213), -1)
        img_resize = cv2.resize(img_current, (640, 360))
        cv2.imshow('blank', img_resize)
        cv2.waitKey(30)
        if np.linalg.norm(np.array(car.car_point) - carInitialPoint1) < 10:
            carInitialPoint = carInitialPoint + np.array(car.car_point)
            carInitialHead = carInitialHead + np.array(car.head_point)
            i += 1
    TimeCount = time.time() - TimeCount
    carInitialPoint = carInitialPoint / 50
    carInitialHead = carInitialHead / 50
    print('fps', j / TimeCount)
    if i / j < 0.8:
        return False, carInitialPoint, carInitialHead
    print('Vision quality is', i / j)
    cv2.destroyAllWindows()
    return True, carInitialPoint, carInitialHead

class TrafficLight:
    def __init__(self):
        self.fd = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.addr = ('192.168.31.163', 6666)
        self.__send_thread = None

    def __del__(self, instance):
        self.fd.close()

    def red(self):
        self.__send(b'red')

    def green(self):
        self.__send(b'green')

    def __run(self,msg):
        for i in range(5):
            self.fd.sendto(msg, self.addr)
            time.sleep(0.1)

    def __send(self,msg):
        if self.__send_thread != None:
            self.__send_thread.join()
        self.__send_thread = threading.Thread(target=self.__run, args=(msg,))
        self.__send_thread.start()


if __name__ == '__main__':
#     car = MiniCar(id = "0000S6")
#     car.set_senseors_upload_time(0.05)
#     car.exec("""
# turn(45)""".encode())
#     while True:
#         #print(car.sensors_data)
#         time.sleep(0.1)
    # init##############################
    cameraNum = 0
    traffic_light = TrafficLight()
    traffic_light.red()
    # 0~1,视觉的置信度
    K_filter = 0.8
    # 转弯中心到后轮的距离（cm）
    lr = 6
    # 转弯中心到前轮的距离（cm）
    lf = 4
    # 得到样板图片
    refer_frame = load_refer_frame()
    cap = open_video(cameraNum)
    slow_planner = trajectoryPlanning()
    fast_planner = trajectoryPlanning_fast()
    parking_plan = trajectoryPlanning_part()
    leaving_plan = trajectoryPlanning_part()
    # 加载红条车
    red_car = RedCar(refer_frame)
    ret, carInitialPoint, carInitialHead = car_init(cap, red_car)
    car_list = []
    if not ret:
        print('Vision system need calibrate')
    # 视觉融合定位
    red_move = carMove(carInitialPoint, carInitialHead, 1, lr, lf, K_filter)
    print('constructing communication with car...')
    red_car_control = MiniCar(id='0000S6')
    # red_car_control.exec(avoid_code)
    car0 = CarControl(0,red_car_control,red_move,red_car,fast_planner,PIDcontroller(0.8, 0, 0),PIDcontroller(0.8, 0, 0.02))
    car_list.append(car0)
    car0.start()

    # # 加载蓝条车
    # blue_car = BlueCar(refer_frame)
    # ret, carInitialPoint, carInitialHead = car_init(cap, blue_car)
    # if not ret:
    #     print('Vision system need calibrate')
    # # 视觉融合定位
    # blue_move = carMove(carInitialPoint, carInitialHead, 1, lr, lf, K_filter)
    # print('constructing communication with car...')
    # blue_car_control = MiniCar(id='0000S6')
    # # blue_car_control.exec(avoid_code)
    # car1 = CarControl(1,blue_car_control,blue_move,blue_car,parking_plan,PIDcontroller(0.8, 0, 0),PIDcontroller(0.8, 0, 0.02))
    # car_list.append(car1)
    # car1.start()
    #
    # # 加载粉条车
    # cyan_car = CyanCar(refer_frame)
    # ret, carInitialPoint, carInitialHead = car_init(cap, cyan_car)
    # if not ret:
    #     print('Vision system need calibrate')
    # # 视觉融合定位
    # cyan_move = carMove(carInitialPoint, carInitialHead, 1, lr, lf, K_filter)
    # print('constructing communication with car...')
    # cyan_car_control = MiniCar(id='0000S2')
    # # cyan_car_control.exec(avoid_code)
    # car2 = CarControl(2,cyan_car_control,cyan_move,cyan_car,slow_planner,PIDcontroller(0.8, 0, 0),PIDcontroller(0.8, 0, 0.02))
    # car_list.append(car2)
    # car2.start()
    #
    # car0.car_list.append(car1)
    # car0.car_list.append(car2)
    # car1.car_list.append(car0)
    # car1.car_list.append(car2)
    # car2.car_list.append(car0)
    # car2.car_list.append(car1)

    stopflag = True
    num = 0
    lasttime = time.time()
    """
    0:fast
    1:slow
    2:parking
    """
    # car_status_list = [1,2,0]
    # car1.status = 3
    miss = 1
    loop = False
    recoder = 0
    while stopflag:
        if time.time() - lasttime > 0.05:
            num += 1
            lasttime = time.time()
            ret, img_current = cap.read()
            for car in car_list:
                car.car.car_info(img_current)
                car.run()
            if car0.status == 0:
                traffic_light.green()
                car0.lap()
                recoder = num
            elif car0.status == 1 and num - recoder > 10:
                traffic_light.red()
            # car0.car.car_info(img_current)
            # car0.run()
            # i = 0
            # while i < 3:
            #     if car_list[i].status != 0 and car_list[i].status != 3:
            #         break
            #     i += 1
            # if i != 3:
            #     continue
            # if loop:
            #     for i in range(len(car_status_list)):
            #         if car_status_list[i] == 2:
            #             if miss == 0:
            #                 car_list[i].leaving(slow_planner)
            #             else:
            #                 car_list[i].leaving(fast_planner)
            #             car_status_list[i] = miss
            #         elif car_status_list[i] == miss:
            #             car_list[i].parking()
            #             car_status_list[i] = 2
            #     loop = False
            #     miss = (miss + 1) & 0x1
            # else:
            #     for car in car_list:
            #         car.lap()
            #     loop = True
            #     car2.car.car_info(img_current)
            #     car2.run()
            
    blue_car_control.disconnect()
    red_car_control.disconnect()
    cyan_car_control.disconnect()
    plt.figure(1, figsize=(6, 6))
    # plt.plot(xhis,yhis)
    plt.scatter(car0._xhis, car0._yhis, alpha=0.2)
    plt.show()

    plt.figure(2, figsize=(6, 6))
    # plt.plot(xhis,yhis)
    plt.scatter(car0._vxhis, car0._vyhis, alpha=0.2)
    plt.show()