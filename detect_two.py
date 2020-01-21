
import numpy as np
import cv2
import logging
import math
import imutils
from time import sleep
import time
np.set_printoptions(threshold=np.inf)
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s: %(message)s')
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
class CarDetection:
    def __init__(self,frame):
        self.kernel = np.ones((9,9),np.uint8)
        self.current_frame = None
        self.refer_frame = frame
        self.last_frame = frame
        self.location = None
        self.block_location = None
        self.direction = None
        self.down_thresh = 40
        self.rect_location = None
        self.car_short = 40
        self.car_long = 80
        self.car_direction = None
        self.location_x = None
        self.location_y = None
        self.location_direction = None
    def pic_process(self,frame):
        gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, self.kernel)
        ret,opening_thresh = cv2.threshold(opening,self.down_thresh,255,0)
        closing = cv2.morphologyEx(opening_thresh, cv2.MORPH_CLOSE, self.kernel)
        ret,closing_thresh = cv2.threshold(closing,self.down_thresh,255,0)
        return closing_thresh
    def flash_last_frame(self):
        diff = cv2.absdiff(self.current_frame,self.last_frame)
        processed_diff = self.pic_process(diff)
        bimg, contours, hier = cv2.findContours(processed_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.imshow("current frame", self.current_frame)
        for cidx,cnt in enumerate(contours):
            AreaRect = cv2.boundingRect(cnt)
            logging.debug("in flash area rect is: %s", AreaRect)
            self.rect_location = AreaRect
            logging.debug("AreaRect is: %s", AreaRect)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            logging.debug("area and perimeter is: %s,%s", area,perimeter)
            if area > 1500 and perimeter < 350:
                roi_left = AreaRect[1] - 20
                roi_right = AreaRect[1] + AreaRect[3] + 20
                roi_up = AreaRect[0] - 20
                roi_down = AreaRect[0]+AreaRect[2] + 20
                self.current_frame[roi_left: roi_right,roi_up:roi_down] = self.last_frame[roi_left : roi_right,roi_up:roi_down]
                self.last_frame = self.current_frame
                break
        else:
            print("flash last frame wrong")
            self.last_frame = self.refer_frame
        # cv2.imshow("last frame", self.last_frame)
        # cv2.waitKey(3000)

    def car_location_detection(self,frame):
        self.current_frame = frame
        cv2.imshow("current",self.current_frame)
        cv2.imshow("last",self.last_frame)
        print(self.current_frame.shape)
        print(self.last_frame)
        cv2.waitKey(10000)
        logging.debug("frame size is: %s", self.current_frame.shape)
        diff = cv2.absdiff(self.current_frame,self.last_frame)
        processed_diff = self.pic_process(diff)
        bimg, contours, hier = cv2.findContours(processed_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # judge size
        for cidx,cnt in enumerate(contours):
            minAreaRect = cv2.minAreaRect(cnt)
            rectCnt = np.int64(cv2.boxPoints(minAreaRect))
            logging.debug("minAreaRect is: %s", minAreaRect)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            logging.debug("area and perimeter is: %s %s", area,perimeter)
            if area > 1500 and perimeter < 350:
                logging.debug("come here")
                self.location = cv2.minAreaRect(cnt)
                logging.debug("car location is: %s", self.location)
                if self.location[1][0] < (self.car_short + self.car_long)/2:
                    self.car_direction = self.location[2] - 90
                else:
                    self.car_direction = self.location[2]
                break
        else:
            logging.warnings("no car found, use the last frame")
    def car_direction_detection(self):
        logging.debug("roi car detection is: %s ",self.location)
        cx = int(self.location[0][0])
        cy = int(self.location[0][1])
        logging.debug("cx and cy is: %s %s",cx,cy)
        roi = self.current_frame[ cy - 50 : cy + 50, cx - 50 : cx + 50]
        cv2.imshow("roi", roi)
        # k = cv2.waitKey(500000)
        
        roi_gray = cv2.cvtColor(roi,cv2.COLOR_RGB2GRAY)
        ret,thresh = cv2.threshold(roi_gray,75,255,1)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel)
        cv2.imshow("thresh", thresh)
        cv2.imshow("closing", closing)
        k = cv2.waitKey(500000)
        if k == 27:
            cv2.destroyAllWindows()
        bimg, contours, hier = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logging.debug("before for")
        logging.debug("direction contours is: %s", len(contours))
        for cidx,cnt in enumerate(contours):
            minAreaRect = cv2.minAreaRect(cnt)
            logging.debug("in car direction, minAreaRect is: %s", minAreaRect)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt,True)
            if area > 400 and perimeter < 130:
                logging.debug("in car direction, y refer is:****************8")
                self.block_location = minAreaRect
            else:
                self.block_location = minAreaRect
                logging.info("no suitable block found, use the last frame")
                pass
            # 1 means up and -1 means down for head 0 means almost horizon
        self.flash_last_frame()
    def car_info(self):
        car_first_line = self.location[1][0]
        car_direction = self.location[2]
        block_cx = self.block_location[0][0]
        block_cy = self.block_location[0][1]
        # means long lines
        if car_first_line > 70:
            if car_direction > -25:
                if block_cx < 50:
                    pass
                else:
                    car_direction -= 180
            else:
                if block_cy < 50:
                    car_direction -= 180
                else:
                    pass
        else:
            if car_direction > -65:
                if block_cy > 50:
                    car_direction -= 90
                else:
                    car_direction -= 270
            else:
                if block_cx < 50:
                    car_direction -= 270
                else:
                    car_direction -= 90
        self.location_x = self.location[0][0]
        self.location_y = self.location[0][1]
        self.location_direction = car_direction
    def car_detection(self,frame):
        self.car_location_detection(frame)
        self.car_direction_detection()
        self.car_info()

class PointRotation:
    def __init__(self):
        self.refer1_point = (95*2,480*2)
        self.refer2_point = (890*2,502*2)
        rotation_angle = math.atan2(self.refer2_point[1] - self.refer1_point[1],\
            self.refer2_point[0] - self.refer1_point[0])
        # for different direction
        self.angle = -rotation_angle
        self.raw_point = None
        self.rotate_point = None
        self.middle_point = None
        self.transform_point = None
    def rotating_coordinate(self,point):
        new_x = point[0] * math.cos(self.angle) + point[1] * math.sin(self.angle)
        new_y = point[1] * math.cos(self.angle) - point[0] * math.sin(self.angle)
        self.rotate_point = (new_x,new_y)
    def move_coordinate(self,point):
        self.middle_point =  (point[0] - self.refer1_point[0],point[1] - self.refer1_point[1])
    def final_coordinate(self,point):
        self.transform_point = (point[0],point[1] + 250)
    def transform(self,point):
        self.raw_point = point
        self.rotating_coordinate(self.raw_point)
        self.move_coordinate(self.raw_point)
        self.final_coordinate(self.middle_point)

class RedCar:
    def __init__(self,frame):
        self.car_location = ((0,0),(80,40),0)
        self.car_direction = 0
        self.car_point = (0,0)
        self.car_pic_rect = (0,0,80,40)
        self.red_block_location = ((0,0),(10,5),0)
        self.refer_frame = frame
        self.current_frame = frame
        self.car_thresh = 60
        self.car_kernel = np.ones((9,9),np.uint8)
        # self.lower_red = np.array([156, 43, 46])
        # self.upper_red = np.array([180, 255, 255])
        self.lower_red = np.array([0, 43, 46])
        self.upper_red = np.array([10, 255, 255])
        self.lower_blue = np.array([100,43,46])
        self.lower_blue = np.array([124,255,255])
        self.red_block_kernel = np.ones((5,5),np.uint8)
        self.red_pixel = 20
        self.red_area = 50
        self.head_point = (0,0)
        self.detection_flag = False
    def location(self):
        # cv2.imshow("refer",self.refer_frame)
        # cv2.imshow("current",self.current_frame)
        # cv2.waitKey(100000)
        # print("####",self.current_frame.shape)
        # print("####",self.last_frame.shape)
        diff = cv2.absdiff(self.current_frame,self.refer_frame)

        gray = cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)
        _,thresh = cv2.threshold(gray,self.car_thresh,255,0)
        open = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.car_kernel)
        # cv2.imshow("diff thresh",thresh)
        # cv2.imshow("diff open",open)
        # k = cv2.waitKey(5000)
        # if k == 27:
        #     cv2.destroyAllWindows()
        car_contours, car_hier = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("%%%%",len(car_contours))
        if len(car_contours) == 0:
            # no car detected
            self.detection_flag = False
        for cidx,cnt in enumerate(car_contours):
            car_x, car_y, car_w, car_h = cv2.boundingRect(cnt)
            self.car_pic_rect = (car_x, car_y, car_w, car_h)
            car = self.current_frame[car_y:car_y+car_h,car_x:car_x+car_w]
            car_hsv = cv2.cvtColor(car, cv2.COLOR_BGR2HSV)
            # process for the red block
            car_mask = cv2.inRange(car_hsv,lowerb=self.lower_red,upperb=self.upper_red)
            red_light = car_mask.sum()
            # cv2.imshow("car",car)
            # cv2.imshow("mask",car_mask)
            # cv2.imshow("open",car_open)
            # cv2.imshow("close",car_close)
            # k = cv2.waitKey(10000)
            # if k == 27:
            #     cv2.destroyAllWindows()
            # at least 10 pixel
            #print("red light is: ",(red_light))
            if red_light > self.red_pixel * 255:
                # print("light is: ",(red_light))
                self.car_location = cv2.minAreaRect(cnt)
                print("car location: ",self.car_location)
                # process the red block area
                # car_open = cv2.morphologyEx(car_mask, cv2.MORPH_OPEN, self.red_block_kernel)
                car_close = cv2.morphologyEx(car_mask, cv2.MORPH_CLOSE, self.red_block_kernel)
                # cv2.imshow("car",car)
                # cv2.imshow("mask",car_mask)
                # # cv2.imshow("open",car_open)
                # cv2.imshow("close",car_close)
                # k = cv2.waitKey(10000)
                # if k == 27:
                #     cv2.destroyAllWindows()
                red_block_contours, red_block_hier = cv2.findContours(car_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for red_block_cidx,red_block_cnt in enumerate(red_block_contours):
                    area = cv2.contourArea(red_block_cnt)
                    if area > self.red_area:
                        self.red_block_location = cv2.minAreaRect(red_block_cnt)
                        print("red block: ", self.red_block_location)
                        self.detection_flag = True
                        break
                else:
                    # no red block found
                    continue
                break
            else:
                # no car detected
                self.detection_flag = False
            
    def car_info(self,frame):
        self.current_frame = frame
        self.location()
        print("red flag is: ", self.detection_flag)
        if self.detection_flag:
            self.car_direction = self.car_location[2]
            self.car_point = self.car_location[0]
            car_area_width = self.car_location[1][0]
            car_area_height = self.car_location[1][1]
            car_pic_rect_width = self.car_pic_rect[2]
            car_pic_rect_height = self.car_pic_rect[3]
            red_block_x = self.red_block_location[0][0]
            red_block_y = self.red_block_location[0][1]
            if car_area_width > car_area_height:
                if self.car_direction > -25:
                    if red_block_x > 0.5 * car_pic_rect_width:
                        pass
                    else:
                        self.car_direction -= 180
                else:
                    if red_block_y < 0.5 * car_pic_rect_height:
                        pass
                    else:
                        self.car_direction -= 180
            else:
                if self.car_direction > -65:
                    if red_block_y < 0.5 * car_pic_rect_height:
                        self.car_direction -= 90
                    else:
                        self.car_direction -= 270
                else:
                    if red_block_x < 0.5 * car_pic_rect_width:
                        self.car_direction -= 90
                    else:
                        self.car_direction -= 270
            car_pi_angel = (-1 * self.car_direction) / 180 * math.pi
            head_x = self.car_point[0] + 50 * math.cos(car_pi_angel)
            head_y = self.car_point[1] - 50 * math.sin(car_pi_angel)
            self.head_point = (head_x,head_y)
        else:
            self.head_point = (-1,-1)
            self.car_point = (-1,-1)

class BlueCar:
    def __init__(self,frame):
        self.car_location = ((0,0),(80,40),0)
        self.car_direction = 0
        self.car_point = (0,0)
        self.car_pic_rect = (0,0,80,40)
        self.blue_block_location = ((0,0),(10,5),0)
        self.refer_frame = frame
        self.current_frame = frame
        self.car_thresh = 60
        self.car_kernel = np.ones((9,9),np.uint8)
        self.lower_red = np.array([156, 43, 46])
        self.upper_red = np.array([180, 255, 255])
        self.lower_blue = np.array([100,43,46])
        self.upper_blue = np.array([124,255,255])
        self.blue_block_kernel = np.ones((3,3),np.uint8)
        self.blue_pixel = 30
        self.blue_area = 30
        self.head_point = (0,0)
        self.detection_flag = False
        # cv2.namedWindow("car")
        # cv2.namedWindow("thresh")
    def location(self):
        # cv2.imshow("refer",self.refer_frame)
        # cv2.imshow("current",self.current_frame)
        # cv2.waitKey(100000)
        diff = cv2.absdiff(self.current_frame,self.refer_frame)
        gray = cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)
        _,thresh = cv2.threshold(gray,self.car_thresh,255,0)
        open = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.car_kernel)
        # cv2.imshow("diff thresh",thresh)
        # cv2.imshow("diff open",open)
        # cv2.waitKey(100000)
        # t = cv2.resize(thresh,(int(1920/2),int(1080/2)))
        # cv2.imshow("thresh", t)
        # k = cv2.waitKey(20)
        # if k == 27:
        #     cv2.destroyAllWindows()
        car_contours, car_hier = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("%%%%",len(car_contours))
        if len(car_contours) == 0:
            # no car detected
            self.detection_flag = False
        for cidx,cnt in enumerate(car_contours):
            car_x, car_y, car_w, car_h = cv2.boundingRect(cnt)
            self.car_pic_rect = (car_x, car_y, car_w, car_h)
            car = self.current_frame[car_y:car_y+car_h,car_x:car_x+car_w]
            car_hsv = cv2.cvtColor(car, cv2.COLOR_BGR2HSV)
            # process for the red block
            car_mask = cv2.inRange(car_hsv,lowerb=self.lower_blue,upperb=self.upper_blue)
            blue_light = car_mask.sum()
            # cv2.imshow("mask",car_mask)
            # cv2.imshow("car",car)
            # k = cv2.waitKey(30)
            # if k == 27:
            #     cv2.destroyAllWindows()
            # at least 10 pixel
            if blue_light > self.blue_pixel * 255:
                # print("light is: ",(red_light))
                self.car_location = cv2.minAreaRect(cnt)
                print("car location: ",self.car_location)
                # process the red block area
                car_open = cv2.morphologyEx(car_mask, cv2.MORPH_OPEN, self.blue_block_kernel)
                car_close = cv2.morphologyEx(car_mask, cv2.MORPH_CLOSE, self.blue_block_kernel)
                # cv2.imshow("car",car)
                # cv2.imshow("mask",car_mask)
                # cv2.imshow("open",car_open)
                # cv2.imshow("close",car_close)
                # cv2.waitKey(20000)
                blue_block_contours, blue_block_hier = cv2.findContours(car_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for blue_block_cidx,blue_block_cnt in enumerate(blue_block_contours):
                    area = cv2.contourArea(blue_block_cnt)
                    if area > self.blue_area:
                        self.blue_block_location = cv2.minAreaRect(blue_block_cnt)
                        print("blue block: ", self.blue_block_location)
                        self.detection_flag = True
                        break
                else:
                    continue
                break
            else:
                # no car detected
                self.detection_flag = False
            
    def car_info(self,frame):
        self.current_frame = frame
        self.location()
        print("blue flag is: ", self.detection_flag)
        if self.detection_flag:
            self.car_direction = self.car_location[2]
            self.car_point = self.car_location[0]
            car_area_width = self.car_location[1][0]
            car_area_height = self.car_location[1][1]
            
            car_pic_rect_width = self.car_pic_rect[2]
            car_pic_rect_height = self.car_pic_rect[3]
            print("car rect width",self.car_pic_rect)
            red_block_x = self.blue_block_location[0][0]
            red_block_y = self.blue_block_location[0][1]
            if car_area_width > car_area_height:
                if self.car_direction > -25:
                    if red_block_x > 0.5 * car_pic_rect_width:
                        pass
                    else:
                        self.car_direction -= 180
                else:
                    if red_block_y < 0.5 * car_pic_rect_height:
                        pass
                    else:
                        self.car_direction -= 180
            else:
                if self.car_direction > -65:
                    if red_block_y < 0.5 * car_pic_rect_height:
                        self.car_direction -= 90
                    else:
                        self.car_direction -= 270
                else:
                    if red_block_x < 0.5 * car_pic_rect_width:
                        self.car_direction -= 90
                    else:
                        self.car_direction -= 270
            car_pi_angel = (-1 * self.car_direction) / 180 * math.pi
            head_x = self.car_point[0] + 50 * math.cos(car_pi_angel)
            head_y = self.car_point[1] - 50 * math.sin(car_pi_angel)
            self.head_point = (head_x,head_y)
        else:
            self.head_point = (-1,-1)
            self.car_point = (-1,-1)


class CyanCar:
    def __init__(self, frame):
        self.car_location = ((0, 0), (80, 40), 0)
        self.car_direction = 0
        self.car_point = (0, 0)
        self.car_pic_rect = (0, 0, 80, 40)
        self.cyan_block_location = ((0, 0), (10, 5), 0)
        self.refer_frame = frame
        self.current_frame = frame
        self.car_thresh = 60
        self.car_kernel = np.ones((9, 9), np.uint8)
        self.lower_cyan = np.array([165, 43, 46])
        self.upper_cyan = np.array([170, 255, 255])
        self.cyan_block_kernel = np.ones((3, 3), np.uint8)
        self.cyan_pixel = 20
        self.cyan_area = 10
        self.head_point = (0, 0)
        self.detection_flag = False

    def location(self):
        # cv2.imshow("refer",self.refer_frame)
        # cv2.imshow("current",self.current_frame)
        # cv2.waitKey(100000)
        diff = cv2.absdiff(self.current_frame, self.refer_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, self.car_thresh, 255, 0)
        open = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.car_kernel)
        # cv2.imshow("diff thresh",thresh)
        # cv2.imshow("diff open",open)
        # cv2.waitKey(100000)
        car_contours, car_hier = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("%%%%", len(car_contours))
        if len(car_contours) == 0:
            # no car detected
            self.detection_flag = False
        for cidx, cnt in enumerate(car_contours):
            car_x, car_y, car_w, car_h = cv2.boundingRect(cnt)
            self.car_pic_rect = (car_x, car_y, car_w, car_h)
            car = self.current_frame[car_y:car_y + car_h, car_x:car_x + car_w]
            car_hsv = cv2.cvtColor(car, cv2.COLOR_BGR2HSV)
            # process for the red block
            car_mask = cv2.inRange(car_hsv, lowerb=self.lower_cyan, upperb=self.upper_cyan)
            cyan_light = car_mask.sum()
            # cv2.imshow("mask",car_mask)
            # cv2.imshow("car",car)
            # cv2.waitKey(20000)
            # at least 10 pixel
            if cyan_light > self.cyan_pixel * 255:
                # print("light is: ",(red_light))
                self.car_location = cv2.minAreaRect(cnt)
                print("car location: ", self.car_location)
                # process the red block area
                car_open = cv2.morphologyEx(car_mask, cv2.MORPH_OPEN, self.cyan_block_kernel)
                car_close = cv2.morphologyEx(car_mask, cv2.MORPH_CLOSE, self.cyan_block_kernel)
                # cv2.imshow("car",car)
                # cv2.imshow("mask",car_mask)
                # cv2.imshow("open",car_open)
                # cv2.imshow("close",car_close)
                # cv2.waitKey(20000)
                cyan_block_contours, cyan_block_hier = cv2.findContours(car_open, cv2.RETR_EXTERNAL,
                                                                        cv2.CHAIN_APPROX_SIMPLE)
                for cyan_block_cidx, cyan_block_cnt in enumerate(cyan_block_contours):
                    area = cv2.contourArea(cyan_block_cnt)
                    if area > self.cyan_area:
                        self.cyan_block_location = cv2.minAreaRect(cyan_block_cnt)
                        print("cyan block: ", self.cyan_block_location)
                        self.detection_flag = True
                        break
                else:
                    continue
                break
            else:
                # no car detected
                self.detection_flag = False

    def car_info(self, frame):
        self.current_frame = frame
        self.location()
        print("cyan flag is: ", self.detection_flag)
        if self.detection_flag:
            self.car_direction = self.car_location[2]
            self.car_point = self.car_location[0]
            car_area_width = self.car_location[1][0]
            car_area_height = self.car_location[1][1]

            car_pic_rect_width = self.car_pic_rect[2]
            car_pic_rect_height = self.car_pic_rect[3]
            print("car rect width", self.car_pic_rect)
            red_block_x = self.cyan_block_location[0][0]
            red_block_y = self.cyan_block_location[0][1]
            if car_area_width > car_area_height:
                if self.car_direction > -25:
                    if red_block_x > 0.5 * car_pic_rect_width:
                        pass
                    else:
                        self.car_direction -= 180
                else:
                    if red_block_y < 0.5 * car_pic_rect_height:
                        pass
                    else:
                        self.car_direction -= 180
            else:
                if self.car_direction > -65:
                    if red_block_y < 0.5 * car_pic_rect_height:
                        self.car_direction -= 90
                    else:
                        self.car_direction -= 270
                else:
                    if red_block_x < 0.5 * car_pic_rect_width:
                        self.car_direction -= 90
                    else:
                        self.car_direction -= 270
            car_pi_angel = (-1 * self.car_direction) / 180 * math.pi
            head_x = self.car_point[0] + 50 * math.cos(car_pi_angel)
            head_y = self.car_point[1] - 50 * math.sin(car_pi_angel)
            self.head_point = (head_x, head_y)
        else:
            self.head_point = (-1, -1)
            self.car_point = (-1, -1)


def refer_init(cam_num):
    cap = cv2.VideoCapture(cam_num,cv2.CAP_DSHOW)
    _ = cap.set(3,1920)
    _ = cap.set(4,1080)
    _ = cap.set(6, cv2.VideoWriter.fourcc('M','J','P','G'))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    for i in range(10):
        _,refer_frame = cap.read()
    cv2.imwrite("blank.jpg",refer_frame)
    cap.release()


def open_video_write(cameraNum,pic_name):
    cap = cv2.VideoCapture(cameraNum, cv2.CAP_DSHOW)
    _ = cap.set(3, 1920)
    _ = cap.set(4, 1080)
    _ = cap.set(6, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS,0)
    _,frame = cap.read()
    cv2.imwrite(pic_name,frame)
    cap.release()
# 1300-1800
# 2300-2800

# 3300-3800
# 4300-4800
if __name__ == "__main__":
    # img_refer = cv2.imread('./test/refer.jpg')
    # img_current = cv2.imread('./test/2686.jpg')
    # cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # _,refer_frame = cap.read()
    # cv2.imwrite("r3.jpg",refer_frame)
    # cap.release()
    # cv2.destroyAllWindows()
    # open_video_write(0,"c3.jpg")
    # # # # test for single pic
    img_refer = cv2.imread('blank.jpg')
    img_current = cv2.imread('c3.jpg')
    # img_refer = cv2.resize(img_refer,(int(1920/2),int(1080/2)))
    # img_current = cv2.resize(img_current,(int(1920/2),int(1080/2)))
    blue_car = CyanCar(img_refer)
    blue_car.car_info(img_current)
    print("middle point: ",blue_car.car_point)
    print("angle",blue_car.car_direction)
    print("head point: ",blue_car.head_point)

    # test blue car
    # from time import sleep
    # import time
    # cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # _,refer_frame = cap.read()
    # cv2.imwrite("blank.jpg",refer_frame)
    # cap.release()
    # # refer_frame = cv2.resize(refer_frame,(int(1920/2),int(1080/2)))
    # red_car = BlueCar(refer_frame)
    # sleep(5)
    # cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    # frame_count = 1
    # while frame_count < 10000:
    #     t1 = time.time()
    #     ret,img_current = cap.read()
    #     # img_current = cv2.resize(img_current,(int(1920/2),int(1080/2)))
    #     # cv2.imshow("frame",img_current)
    #     # print("come")
    #     red_car.car_info(img_current)
    #     # k = cv2.waitKey(30)
    #     # if k == 27:
    #     #     break
    #     print("middle point: ",red_car.car_point)
    #     print("angle",red_car.car_direction)
    #     print("head point: ",red_car.head_point)
    #     frame_count += 1
    #     t2 = time.time()
    #     print("time interval is: ", t2-t1)
    # test red car and blue car



# self test
    # cap = cv2.VideoCapture(0)
    # _,refer_frame = cap.read()
    # cv2.imwrite("blank.jpg",refer_frame)
    # cap.release()
    # # refer_frame = cv2.resize(refer_frame,(int(1920/2),int(1080/2)))
    # red_car = RedCar(refer_frame)
    # blue_car = BlueCar(refer_frame)
    # sleep(5)
    # cap = cv2.VideoCapture(0)
    # frame_count = 1
    # while frame_count < 10000:
    #     t1 = time.time()
    #     ret,img_current = cap.read()
    #     # img_current = cv2.resize(img_current,(int(1920/2),int(1080/2)))
    #     cv2.imshow("frame",img_current)
    #     # print("come")
    #     red_car.car_info(img_current)
    #     blue_car.car_info(img_current)
    #     k = cv2.waitKey(30)
    #     if k == 27:
    #         break
    #     print("****middle point:**** ",red_car.car_point)
    #     print("******angle*****",red_car.car_direction)
    #     print("*****head point:***** ",red_car.head_point)
    #     print("$$$$$middle$$$$",blue_car.car_point)
    #     print("$$$$$$$*ngle$$$$$",blue_car.car_direction)
    #     print("$$$$$head point:$$$$$ ",blue_car.head_point)
    #     frame_count += 1
    #     t2 = time.time()
    #     print("time interval is: ", t2-t1)












'''
    diff = cv2.absdiff(img_current,img_refer)
    img_resize = cv2.resize(diff,(int(1920),int(1080)))
    gray = cv2.cvtColor(img_resize,cv2.COLOR_RGB2GRAY)
    _,thresh = cv2.threshold(gray,10,255,0) 
    kernel = np.ones((9,9),np.uint8)
    open = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # closing = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, hier = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    for cidx,cnt in enumerate(contours):
        line_kernel = np.ones((5,5),np.uint8)
        x, y, w, h = cv2.boundingRect(cnt)
        print("fds",cv2.boundingRect(cnt))
        roi = img_current[y:y+h,x:x+w]
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        low_hsv = np.array([156,100,100])
        high_hsv = np.array([180,255,255])
        roi_mask = cv2.inRange(roi_hsv,lowerb=low_hsv,upperb=high_hsv)
        light = roi_mask.sum()
        if light > 10 * 255:
            print("light is: ",(light))
            car_location = cv2.minAreaRect(cnt)
            print("car location: ",car_location)
            roi_open = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, line_kernel)
            roi_close = cv2.morphologyEx(roi_open, cv2.MORPH_CLOSE, line_kernel)
            cv2.imshow("mask",roi_mask)
            cv2.imshow("open",roi_open)
            cv2.imshow("close",roi_close)
            roi_contours, roi_hier = cv2.findContours(roi_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cidx,roi_cnt in enumerate(roi_contours):

                roi_rect = cv2.minAreaRect(roi_cnt)
                roi_box = cv2.cv.Boxpoints(roi_rect) if imutils.is_cv2() else cv2.boxPoints(roi_rect)
                roi_box = np.int0(roi_box)
                print("box is: ",roi_rect)
                cv2.drawContours(roi, [roi_box], 0, (0, 0, 255), 2)
                cv2.imshow("roi",roi)
                # cv2.imshow("d",diff_r)
                k = cv2.waitKey(1000000)
                if k == 27:
                    cv2.destroyAllWindows()
                else:
                    continue

    




    # first image
    img_current = cv2.imread('2446.jpg')
    car.car_detection(img_current)
    pixel_width = 300/1920
    pixel_height = 150/1080
    x_location = pixel_width * car.location_x
    y_location = pixel_height * car.location_y
    print("car location is: ",x_location  ,y_location)
    print("car direction is: ",car.location_direction)
    location_point = (x_location,y_location)
    point_rotation.transform((890*2,502*2))
    print("real world location is: ", point_rotation.transform_point)
    pixel_width = 300/1920
    pixel_height = 150/1080
    x_location = car.location[0][0] * pixel_width
    y_location = car.location[0][1] * pixel_height
    print("&&&&&&&&&",x_location,y_location)
    print("direction: ",car.car_direction)


    image_stitch = Image_Stitching()
    up_img1 = cv2.imread("./1220dataset/1301.jpg")
    down_img1 = cv2.imread("./1220dataset/2301.jpg")
    cv2.imshow("up",up_img1)
    cv2.imshow("down",down_img1)
    k = cv2.waitKey(100000)
    if k == 27:
        cv2.destroyAllWindows()


    # current_direction_detection1(img_current)
    img_refer = cv2.imread('1223.jpg')
    car = CarDetection(img_refer)
    # first image
    img_current = cv2.imread('1190.jpg')
    car.car_detection(img_current)
    print("location:",car.location)
    print("direction: ",car.direction)
    # second image
    img_next = cv2.imread("1193.jpg")
    # cv2.imshow("next",img_next)
    # cv2.waitKey(1000)
    car.car_detection(img_next)
    print("location:",car.location)
    print("direction: ",car.direction)
    # third image
    img_next = cv2.imread("1074.jpg")
    car.car_detection(img_next)
    print("location:",car.location)
    print("direction: ",car.direction)
'''
