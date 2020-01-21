import numpy as np
import cv2
import logging
import math
import imutils
from time import sleep
import time

np.set_printoptions(threshold=np.inf)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s: %(message)s')


# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

class RedCars:
    def __init__(self, frame):
        self.car_location = ((0, 0), (80, 40), 0)
        self.car_direction = 0
        self.car_point = (0, 0)
        self.car_pic_rect = (0, 0, 80, 40)
        self.red_block_location = ((0, 0), (10, 5), 0)
        frame = frame[:, 500:]
        self.refer_frame = frame
        self.current_frame = frame
        self.car_thresh = 70
        self.car_kernel = np.ones((9, 9), np.uint8)
        self.lower_red = np.array([175, 43, 46])
        self.upper_red = np.array([180, 255, 255])
        self.lower_red1 = np.array([0, 43, 46])
        self.upper_red1 = np.array([4, 255, 255])
        self.red_block_kernel = np.ones((3, 3), np.uint8)
        self.red_pixel = 20
        # self.red_area = 10
        # self.red_pixel = 40
        self.red_area = 50
        self.head_point = (0, 0)
        self.detection_flag = False
        self.car_locations = []
        self.car_pic_rects = []
        self.red_block_locations = []
        self.detection_flags = []
        self.car_points = []
        self.head_points = []
        self.red_car0_point = [0, 0]
        self.red_car0_head_point = [0, 0]
        self.red_car1_point = [0, 0]
        self.red_car1_head_point = [0, 0]
        self.red_car2_point = [0, 0]
        self.red_car2_head_point = [0, 0]
        self.distance_erro = 0

    def location(self):
        # cv2.imshow("refer",self.refer_frame)
        # cv2.imshow("current",self.current_frame)
        # k = cv2.waitKey(100000)
        # if k == 27:
        #     cv2.destroyAllWindows()
        diff = cv2.absdiff(self.current_frame, self.refer_frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, self.car_thresh, 255, 0)
        open = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.car_kernel)
        # cv2.imshow("diff thresh",thresh)
        # cv2.imshow("diff open",cv2.resize(open,(640,480)))
        # k = cv2.waitKey(30)
        # if k == 27:
        #     cv2.destroyAllWindows()
        car_contours, car_hier = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print("%%%%", len(car_contours))
        for cidx, cnt in enumerate(car_contours):
            car_x, car_y, car_w, car_h = cv2.boundingRect(cnt)
            self.car_pic_rect = (car_x, car_y, car_w, car_h)
            # print("car pic rect",self.car_pic_rect)
            car = self.current_frame[car_y:car_y + car_h, car_x:car_x + car_w]
            car_hsv = cv2.cvtColor(car, cv2.COLOR_BGR2HSV)
            # process for the red block
            car_mask1 = cv2.inRange(car_hsv, lowerb=self.lower_red, upperb=self.upper_red)
            car_mask2 = cv2.inRange(car_hsv, lowerb=self.lower_red1, upperb=self.upper_red1)
            car_mask = car_mask1 | car_mask2
            red_light = car_mask.sum()
            # print("light is: ",(red_light))
            # cv2.imshow("car",car)
            # k = cv2.waitKey(100000)
            # if k == 27:
            #     cv2.destroyAllWindows()
            # at least 10 pixel
            if red_light > self.red_pixel * 255:
                # print("light is: ",(red_light))
                self.car_location = cv2.minAreaRect(cnt)
                print("car location: ", self.car_location)
                # process the red block area
                # car_open = cv2.morphologyEx(car_mask, cv2.MORPH_OPEN, self.red_block_kernel)
                car_close = cv2.morphologyEx(car_mask, cv2.MORPH_CLOSE, self.red_block_kernel)
                # cv2.imshow("car hsv",car_hsv)
                # cv2.imshow("mask",car_mask)
                # cv2.imshow("open",car_open)
                # cv2.imshow("close",car_close)
                # k = cv2.waitKey(100000)
                # if k == 27:
                #     cv2.destroyAllWindows()
                red_block_contours, red_block_hier = cv2.findContours(car_close, cv2.RETR_EXTERNAL,
                                                                      cv2.CHAIN_APPROX_SIMPLE)
                for red_block_cidx, red_block_cnt in enumerate(red_block_contours):
                    area = cv2.contourArea(red_block_cnt)
                    if area > self.red_area:
                        self.red_block_location = cv2.minAreaRect(red_block_cnt)
                        # print("red block: ", self.red_block_location)
                        red_width = self.red_block_location[1][0]
                        red_height = self.red_block_location[1][1]
                        if red_width > 10 and red_height > 10:
                            # box = cv2.BoxPoints(self.red_block_location)
                            box = cv2.cv.Boxpoints() if imutils.is_cv2() else cv2.boxPoints(self.red_block_location)

                            box = np.int0(box)
                            # cv2.drawContours(car, [box], 0, (255, 0, 0), 1)
                            # show red block
                            # cv2.imshow("red block",car)
                            # k = cv2.waitKey(100000)
                            # if k == 27:
                            #     cv2.destroyAllWindows()
                            self.detection_flag = True
                            self.car_locations.append(self.car_location)
                            self.car_pic_rects.append(self.car_pic_rect)
                            self.red_block_locations.append(self.red_block_location)
                        else:
                            print("invalid red block, ingore this")
                self.detection_flags.append(self.detection_flag)
                # print(self.detection_flags)

    def car_calculate(self):
        # init the values
        self.car_locations = []
        self.car_pic_rects = []
        self.red_block_locations = []
        self.detection_flags = []
        # compelete car detection
        self.location()
        # calculate the car nums
        true_num = sum(self.detection_flags)
        print(true_num)
        try:
            if true_num == 3:
                self.car_points = []
                self.head_points = []
                for i in range(true_num):
                    self.car_direction = self.car_locations[i][2]
                    self.car_point = self.car_locations[i][0]
                    car_area_width = self.car_locations[i][1][0]
                    car_area_height = self.car_locations[i][1][1]
                    car_pic_rect_width = self.car_pic_rects[i][2]
                    car_pic_rect_height = self.car_pic_rects[i][3]
                    red_block_x = self.red_block_locations[i][0][0]
                    red_block_y = self.red_block_locations[i][0][1]
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
                    head_x = self.car_point[0] + 30 * math.cos(car_pi_angel)
                    head_y  = self.car_point[1] - 30 * math.sin(car_pi_angel)
                    self.head_point = (head_x, head_y)
                    self.head_points.append(self.head_point)
                    self.car_points.append(self.car_point)
            else:
                self.car_points = []
                self.head_points = []
                logging.info("detection numbers is not right")
                self.head_point = (-1, -1)
                self.car_point = (-1, -1)
                self.head_points.append(self.head_point)
                self.car_points.append(self.car_point)
        except:
            self.car_points = []
            self.head_points = []
            logging.info("detection numbers is not right")
            self.head_point = (-1, -1)
            self.car_point = (-1, -1)
            self.head_points.append(self.head_point)
            self.car_points.append(self.car_point)
        print("&&&", self.car_points)
        print("$$$$", self.head_points)

    def car_index_init(self, frame):
        self.current_frame = frame[:,500:]
        self.car_calculate()
        # according to x corrdinate
        if len(self.head_points) == 3:
            self.car_points.sort()
            self.head_points.sort()
            self.red_car0_point = self.car_points[0]
            self.red_car1_point = self.car_points[1]
            self.red_car2_point = self.car_points[2]
            self.red_car0_head_point = self.head_points[0]
            self.red_car1_head_point = self.head_points[1]
            self.red_car2_head_point = self.head_points[2]
            self.car_points = []
            self.head_points = []
        else:
            print("car index initializa error")
            self.car_points = []
            self.head_points = []
            return -1

    def car_distance(self, first, second):
        distance_error = abs(first[0] - second[0]) + abs(first[1] - second[1])
        return distance_error

    def car_info(self, frame):
        self.current_frame = frame[:,500:]
        self.car_calculate()
        # self.red_car0_head_point - self.
        # print("_____",self.red_car0_point)
        # print("!!!!!",self.car_points)
        if len(self.head_points) == 3:
            for i in range(len(self.head_points)):
                # print("~~~~",self.car_points[i],self.red_car0_point)
                dis0 = self.car_distance(self.red_car0_point, self.car_points[i])
                dis1 = self.car_distance(self.red_car1_point, self.car_points[i])
                dis2 = self.car_distance(self.red_car2_point, self.car_points[i])
                # print("minimin",dis0,dis1,dis2)
                if min(dis0, dis1, dis2) == dis0:
                    self.red_car0_point = self.car_points[i]
                    self.red_car0_head_point = self.head_points[i]
                    self.red_car0_location = self.car_locations[i]
                if min(dis0, dis1, dis2) == dis1:
                    self.red_car1_point = self.car_points[i]
                    self.red_car1_head_point = self.head_points[i]
                    self.red_car1_location = self.car_locations[i]
                if min(dis0, dis1, dis2) == dis2:
                    self.red_car2_point = self.car_points[i]
                    self.red_car2_head_point = self.head_points[i]
                    self.red_car2_location = self.car_locations[i]
            for i in range(3):
                if self.red_car0_point == self.car_locations[i][0]:
                    box = cv2.cv.Boxpoints() if imutils.is_cv2() else cv2.boxPoints(self.car_locations[i])
                    box = np.int0(box)
                    cv2.drawContours(self.current_frame, [box], 0, (255,0,0), 4)
                if self.red_car1_point == self.car_locations[i][0]:
                    box = cv2.cv.Boxpoints() if imutils.is_cv2() else cv2.boxPoints(self.car_locations[i])
                    box = np.int0(box)
                    cv2.drawContours(self.current_frame, [box], 0, (0,255,0), 4)
                if self.red_car2_point == self.car_locations[i][0]:
                    box = cv2.cv.Boxpoints() if imutils.is_cv2() else cv2.boxPoints(self.car_locations[i])
                    box = np.int0(box)
                    cv2.drawContours(self.current_frame, [box], 0, (0,0,255), 4)
        else:
            return -1


def refer_init():
    cap = cv2.VideoCapture(0)
    _, refer_frame = cap.read()
    cv2.imwrite("blank.jpg", refer_frame)
    cap.release()


# 1300-1800
# 2300-2800

# 3300-3800
# 4300-4800
if __name__ == "__main__":
    # img_refer = cv2.imread('./test/refer.jpg')
    # img_current = cv2.imread('./test/2686.jpg')
    # cap = cv2.VideoCapture(0)
    # _,refer_frame = cap.read()
    # cv2.imwrite("ttt1.jpg",refer_frame)
    # cap.release()
    # test fofr single pic
    # vid = cv2.VideoCapture("record3.avi")
    # for i in range(1000):
    #     _,frame = vid.read()
    #     pic_name = str(i) + "_t.jpg"
    #     if i > 100 and i < 1000 and i % 100 == 10:
    #         cv2.imwrite(pic_name,frame)

    # 262, 262, 294, 262, 349, 330, 262, 262, 294, 262, 392, 349, 262, 262, 523, 440, 349, 330, 294, 466, 466, 440, 349, 392, 349
    # 0.5, 0.5, 1, 1, 1, 2, 0.5, 0.5, 1, 1, 1, 2, 0.5, 0.5, 1, 1, 1, 1, 3, 0.5, 0.5, 1, 1, 1, 2
    # vs = cv2.VideoCapture("temp.mov")
    # # _,img_refer = vs.read()
    # img_refer = cv2.imread('one_refer.jpg')
    # red_car = RedCars(img_refer)

    # _,img_init = vs.read()
    # red_car.car_index_init(img_init)
    # img_refer = cv2.resize(img_refer,(int(1920/2),int(1080/2)))
    # img_init = cv2.resize(img_init,(int(1920/2),int(1080/2)))

    # red_car.car_info(img_current)

    img_refer = cv2.imread('blank.jpg')
    img_current = cv2.imread('410_t.jpg')
    img_refer = cv2.resize(img_refer, (int(1920 / 2), int(1080 / 2)))
    # img_new = cv2.imread('1_t.jpg')
    img_current = cv2.resize(img_current, (int(1920 / 2), int(1080 / 2)))
    red_car = RedCars(img_refer)
    red_car.car_info(img_current)

    # # car_hsv = cv2.cvtColor(img_current, cv2.COLOR_BGR2HSV)
    # # cv2.imshow("fdsfsd",car_hsv)
    # # k = cv2.waitKey(70000)
    # # if k == 27:
    # #     cv2.destroyAllWindows()
    # for i in range(2000):
    #     _,img_new = vs.read()
    #     # img_new = cv2.resize(img_new,(int(1920/2),int(1080/2)))
    #     print("ready to detected")
    #     red_car.car_info(img_new)
    #     print("After detected")
    #     red_car0_head_point = tuple([int(x) for x in red_car.red_car0_head_point])
    #     red_car1_head_point = tuple([int(x) for x in red_car.red_car1_head_point])
    #     red_car2_head_point = tuple([int(x) for x in red_car.red_car2_head_point])
    #     red_car0_point = tuple([int(x) for x in red_car.red_car0_point])
    #     red_car1_point = tuple([int(x) for x in red_car.red_car1_point])
    #     red_car2_point = tuple([int(x) for x in red_car.red_car2_point])
    #     print("&&&&&&red_car0&&&&&&",red_car0_head_point,red_car0_point)
    #     cv2.line(img_new,red_car0_head_point,red_car0_point,(0,0,255),1,8)
    #     cv2.line(img_new,red_car1_head_point,red_car1_point,(0,255,0),1,8)
    #     cv2.line(img_new,red_car2_head_point,red_car2_point,(255,0,0),1,8)
    #     cv2.imshow("tracking",img_new)
    #     k = cv2.waitKey(30)
    #     if k == 27:
    #         cv2.destroyAllWindows()
    #         break
    # print(red_car.red_car0_point,red_car.red_car1_point,red_car.red_car2_point)

    # test blue car
    # from time import sleep
    # import time
    # cap = cv2.VideoCapture(0)
    # _,refer_frame = cap.read()
    # cv2.imwrite("blank.jpg",refer_frame)
    # cap.release()
    # refer_frame = cv2.resize(refer_frame,(int(1920/2),int(1080/2)))
    # red_car = BlueCar(refer_frame)
    # sleep(5)
    # cap = cv2.VideoCapture(1)
    # frame_count = 1
    # while frame_count < 10000:
    #     t1 = time.time()
    #     ret,img_current = cap.read()
    #     # img_current = cv2.resize(img_current,(int(1920/2),int(1080/2)))
    #     cv2.imshow("frame",img_current)
    #     # print("come")
    #     red_car.car_info(img_current)
    #     k = cv2.waitKey(30)
    #     if k == 27:
    #         break
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
