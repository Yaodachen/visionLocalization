import time
import socket
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from detect_two_RBGY import RedCars
from detect_two import refer_init
from mini_car_23 import MiniCar, carMove
from trajectoryPlanning import trajectoryPlanning,trajectoryPlanning_fast,trajectoryPlanning_part,lap1,lap2
from PIDcontroller import PIDcontroller
import threading

two_camera_flag = 0

class CarControl:
    __park_code = """
turn(35)
go(-20,1.8)
turn(0)
go(-15,1)""".encode()

    __turn_around_code = """
turn(35)
go(-50,1.2)
turn(-35)
go(50,1)""".encode()

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
        # ???????????cm?
        self._lr = 6
        # ???????????cm?
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
        sigma = self._ang * 3.14 / (1.5 * 180)  # ????????????????1.5 : 1
        beta = np.arctan(self._lr / (self._lf + self._lr) * np.tan(sigma)) / 1.2  # ???????????
        #?????
        head_point_list = [list([int(x) for x in red_car.red_car0_head_point]),list([int(x) for x in red_car.red_car1_head_point]),list([int(x) for x in red_car.red_car2_head_point])]
        car_point_list = [list([int(x) for x in red_car.red_car0_point]),list([int(x) for x in red_car.red_car1_point]), list([int(x) for x in red_car.red_car2_point])]
        for p in head_point_list:
            p[0] += 500
        for p in car_point_list:
            p[0] += 500

        vision = self.move.getVisionPosition(car_point_list[self.car_num], head_point_list[self.car_num], 1)
        xv = vision[0]
        yv = vision[1]
        if np.linalg.norm([self._xhis[-1] - xv,self._yhis[-1] - yv]) > 50:
            location_mode = 0  # ??????
        else:
            location_mode = 1  # ????
        xw = self.move.getCarPosition(car_point_list[self.car_num], head_point_list[self.car_num], 1, v, beta, dt,
                                 location_mode)  # 3????x,y,phi
        phiVision = math.atan2(head_point_list[self.car_num][1] - car_point_list[self.car_num][1],
                               head_point_list[self.car_num][0] - car_point_list[self.car_num][0])
        # xw = [xv,yv,phiVision]
        self._dthis.append(dt)
        self._visionCarPointHis.append(car_point_list[self.car_num])
    
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
                if self.is_park and self.status == 2:
                    self.car_control.exec(self.__park_code)
                    self.is_park = False
                    self.status = 3
                elif self.status == 4 or self.status == 1:
                    self.status = 0
        if self.status == 5 and self.planner.nowPoint == 2:
            traffic_light.red()
            self.car_control.exec(self.__turn_around_code)
            self.planner.speed = 100
            self.planner.destinatePoint += 1
            self.planner.nowPoint += 1
            self.planner.newFlag = 1

        speed = self._speed
        ang = self._ang

        space = self.__car_min_space()
        # if self.status == 1 and space < 40 and 1 < self.planner.nowPoint < 6 and car_status_list[self.car_num] == 1:
        #     p = space / 50 + 0.2
        #     speed = self._speed * p
        if self.car_location[1] > 155:
            if self.car_num == 0:
                if self.car_control.get_distance()[1] < 100:
                    speed = 0
                    self.planner.beginTime += 0.05
                else:
                    speed = 10
            elif self.car_num == 2:
                if self.car_control.get_distance()[0] < 100:
                    speed = 0
                    self.planner.beginTime += 0.05
                else:
                    speed = 10
        else:
            if speed < 20:
                speed = 20
        print(self.car_num, ' status:', self.status)
        print(self.car_num, ' car_space:', space)
        print(self.car_num,' distance:', self.car_control.get_distance())
        print(self.car_num,' speed,ang:', speed,ang)
        # if self.car_num == 1 and (self.planner.nowPoint > 5 or is_end) and self.car_location[0] < 200:
        #     speed = 0
        if speed > 30:
            speed = 30
        if ang > 35:
            ang = 35
        elif ang < -35:
            ang = -35
        if self.status == 0:
            speed = 0
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
        if self.status == 5:
            self.__parking()
            self.status = 2

    def red_park(self):
        if self.status == 0:
            self.status = 5
            red_park_plan.landmark = np.array([[180, 235], [150, 225], [150, 60], [170, 200], [200, 200]])
            self.planner = red_park_plan
            self.planner.start()

    def __parking(self):
        # if self.status == 1:
        #     parking_plan.landmark = np.array([self.__my_planner.landmark[0],[136,225],[35,225],[35,145]])
        #     self.planner = parking_plan
        #     self.planner.start()
        #     self.is_park = True
        # elif self.status == 0:
        parking_plan.landmark = np.array([[160, 215], [35, 215], [35, 100]])
        self.planner = parking_plan
        self.planner.start()
        self.is_park = True

    def leaving(self):
        if self.status == 3:
            self.status = 4
            leaving_plan.landmark = np.array([[25,137],[25,245],[160,235],[200,200]])
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

# ????????????cameraNum??
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

def load_car_frame(cam_num = 0):
    cap = cv2.VideoCapture(cam_num,cv2.CAP_DSHOW)
    _ = cap.set(3,1920)
    _ = cap.set(4,1080)
    _ = cap.set(6, cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    for i in range(10):
        _,refer_frame = cap.read()
    cap.release()
    cv2.imshow('blank',cv2.resize(refer_frame, (640, 360)))
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
    if two_camera_flag == 1:
        cap2 = cv2.VideoCapture(cameraNum, cv2.CAP_DSHOW)
        _ = cap2.set(3, 1920)
        _ = cap2.set(4, 1080)
        _ = cap2.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        cap2.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap2.set(cv2.CAP_PROP_FOCUS, 0)
        return cap,cap2
    return cap


# ?????
def car_init(cap, car):
    carPointList = np.zeros(shape=(3, 50, 2))
    headPointList = np.zeros(shape=(3, 50, 2))
    carInitialPoint = np.zeros(shape=(3, 2))
    carInitialPoint1 = np.zeros(shape=(3, 2))
    carInitialHead = np.zeros(shape=(3, 2))
    i = 0
    while i < 50:
        ret, img_current = cap.read()
        car.car_info(img_current)
        head_point_list = [list([int(x) for x in red_car.red_car0_head_point]),list([int(x) for x in red_car.red_car1_head_point]),list([int(x) for x in red_car.red_car2_head_point])]
        car_point_list = [list([int(x) for x in red_car.red_car0_point]),list([int(x) for x in red_car.red_car1_point]), list([int(x) for x in red_car.red_car2_point])]
        for p in head_point_list:
            p[0] += 500
        for p in car_point_list:
            p[0] += 500
        for index in range(3):
            car_point = car_point_list[index]
            if car_point[0] > 0 and car_point[1] > 0:
                carPointList[index,i, :] = np.array(car_point_list[index])
                headPointList[index,i, :] = np.array(head_point_list[index])
        i += 1

    for index in range(3):
        carInitialPoint1[index,:] = [np.median(carPointList[index,:, 0]), np.median(carPointList[index,:, 1])]
    j = 0
    i = [0,0,0]
    TimeCount = time.time()
    while i[0] < 50:
        j += 1
        ret, img_current = cap.read()
        car.car_info(img_current)
        head_point_list = [list([int(x) for x in red_car.red_car0_head_point]),list([int(x) for x in red_car.red_car1_head_point]),list([int(x) for x in red_car.red_car2_head_point])]
        car_point_list = [list([int(x) for x in red_car.red_car0_point]),list([int(x) for x in red_car.red_car1_point]), list([int(x) for x in red_car.red_car2_point])]
        print(j, "before car_point", car_point_list)
        for p in head_point_list:
            p[0] += 500
        for p in car_point_list:
            p[0] += 500
        print(j,"car_point",car_point_list)
        for index in range(3):
            cv2.circle(img_current, (int(car_point_list[index][0]), int(car_point_list[index][1])), 6, (213, 213, 0), -1)
            cv2.circle(img_current, (int(head_point_list[index][0]), int(head_point_list[index][1])), 6, (255, 0, 0), -1)
        img_resize = cv2.resize(img_current, (640, 360))
        cv2.imshow('blank', img_resize)
        cv2.waitKey(30)
        for index in range(3):
            if np.linalg.norm(np.array(car_point_list[index]) - carInitialPoint1[index]) < 10:
                carInitialPoint[index] = carInitialPoint[index] + np.array(car_point_list[index])
                carInitialHead[index] = carInitialHead[index]  + np.array(head_point_list[index])
                i[index] += 1
    TimeCount = time.time() - TimeCount
    for index in range(3):
        carInitialPoint[index] = carInitialPoint[index] / i[index]
        carInitialHead[index] = carInitialHead[index] / i[index]
    print('fps', j / TimeCount)
    if i[0] / j < 0.8:
        return False, carInitialPoint, carInitialHead
    print('Vision quality is', i[0] / j)
    cv2.destroyAllWindows()
    return True, carInitialPoint, carInitialHead

class TrafficLight:
    def __init__(self):
        self.fd = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.addr = ('192.168.31.163', 6666)
        self.__send_thread = None

    def __del__(self):
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
    # cv2.imshow('test',cv2.imread('blank.jpg')[:,500:])
    # cv2.waitKey(100000)
    traffic_light = TrafficLight()
    traffic_light.green()
    # 0~1,??????
    K_filter = 1
    # ???????????cm?
    lr = 6
    # ???????????cm?
    lf = 4
    # ??????
    refer_frame = load_refer_frame()
    slow_planner = trajectoryPlanning()
    fast_planner = trajectoryPlanning_fast()
    parking_plan = trajectoryPlanning_part()
    leaving_plan = trajectoryPlanning_part()
    red_park_plan = trajectoryPlanning_part()
    lap1_plan = lap1()
    lap2_plan = lap2()
    left_plan = trajectoryPlanning_part()
    left_plan.landmark = np.array([[155,157],[150,140],[147,120],[123,60]])
    left_plan.speed = 10
    mid_plan = trajectoryPlanning_part()
    mid_plan.landmark = np.array([[147,140],[147,130],[147,30]])
    mid_plan.speed = 30
    right_plan = trajectoryPlanning_part()
    right_plan.landmark = np.array([[140,157],[147,140],[147,130],[147,100],[160,60]])
    right_plan.speed = 10
    #?????
    red_car = RedCars(refer_frame)
    red_car.car_index_init(load_car_frame())#?????,x??????
    cap = open_video(cameraNum)
    ret, carInitialPoint, carInitialHead = car_init(cap,red_car)
    car_list = []
    #No.0
    car0_control = MiniCar(id='0000S3')
    car0 = CarControl(0,car0_control,carMove(carInitialPoint[0], carInitialHead[0], 1, lr, lf, K_filter),red_car,left_plan,PIDcontroller(0.8, 0, 0),PIDcontroller(0.8, 0, 0.02))
    car_list.append(car0)
    # No.1
    car1_control = MiniCar(id='0000RT')
    car1 = CarControl(1, car1_control, carMove(carInitialPoint[1], carInitialHead[1], 1, lr, lf, K_filter), red_car,
                  mid_plan, PIDcontroller(0.8, 0, 0), PIDcontroller(0.8, 0, 0.02))
    car_list.append(car1)
    # No.2
    car2_control = MiniCar(id='0000S6')
    car2 = CarControl(2, car2_control, carMove(carInitialPoint[2], carInitialHead[2], 1, lr, lf, K_filter), red_car,
                  right_plan, PIDcontroller(0.8, 0, 0), PIDcontroller(0.8, 0, 0.02))
    car_list.append(car2)
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
    car_status_list = [2,1,0]
    miss = 1
    loop = False
    recoder = 0

    for car in car_list:
        car.start()

    _ = cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    fps = 30
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('video' + str(int(time.time()))+'.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),fps, size)

    while stopflag:
        if time.time() - lasttime > 0.05:
            num += 1
            lasttime = time.time()
            ret, img_current = cap.read()
            if two_camera_flag == 1:
                ret, img_current2 = cap2.read()
                red_car.car_info(img_current)
                head_point_list = [tuple([int(x) for x in red_car.red_car0_head_point]),
                                   tuple([int(x) for x in red_car.red_car1_head_point]),
                                   tuple([int(x) for x in red_car.red_car2_head_point])]
                car_point_list = [tuple([int(x) for x in red_car.red_car0_point]),
                                  tuple([int(x) for x in red_car.red_car1_point]),
                                  tuple([int(x) for x in red_car.red_car2_point])]
            else:
                red_car.car_info(img_current)
            for car in car_list:
                car.run()
            if num == 10:
                car1.lap()
                car0.lap()
                car2.lap()
            if num == 150:
                car0.status = 0
            cv2.imshow('now',cv2.resize(img_current,(640,360)))
            if cv2.waitKey(10) == 27:
                break
            print(num)
            videoWriter.write(img_current)
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
            
    car0_control.disconnect()
    car1_control.disconnect()
    car2_control.disconnect()
    plt.figure(1, figsize=(6, 6))
    # plt.plot(xhis,yhis)
    plt.scatter(car1._xhis, car1._yhis, alpha=0.2)
    plt.show()

    # plt.figure(2, figsize=(6, 6))
    # # plt.plot(xhis,yhis)
    # plt.scatter(car0._vxhis, car0._vyhis, alpha=0.2)
    plt.show()