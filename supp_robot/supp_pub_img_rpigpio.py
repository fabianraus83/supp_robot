#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 11:16:21 2021

@author: fabian
"""

import cv2
import threading
#import signal

import time
import numpy as np

#from PIL import Image as PILImage
#from yolov4 import Detector

import rclpy
from rclpy.node import Node
#from std_msgs.msg import String
from cv_bridge import CvBridge
#from sensor_msgs.msg import Image
from pic_message.msg import SupportImage

import RPi.GPIO as GPIO


GPIO.setmode(GPIO.BCM)

GPIO_TRIGECHO = 10

print("Ultrasonic Measurement")

# Set pins as output and input
GPIO.setup(GPIO_TRIGECHO, GPIO.OUT)
print("declar output GPIO as GPIO{}".format(GPIO_TRIGECHO))

path = "/Desktop/flask_sqlite_html/multithread_images/"
picname = "thread.jpg"
robot_id = int(input("1 or 2"))
dist = 0

i = 0 #iteration number in main loop

people_dict = {}

cam_IDs = (0,2)


class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.frame = None
#        self.daemon = True
        self.rval = False
        self.write = False
        self.end = False
        self.j = 0
        self.savelocation = "" 
#        self.cam = cv2.VideoCapture(self.camID)
#        cam = cv2.VideoCapture(self.camID)
#        self.cam = cam
        
    def run(self):
        print("camID is " + str(self.camID))
        print("initiate run")
        print("named window")
        cam = cv2.VideoCapture(self.camID)
        print("before if else loop")
        if cam.isOpened():
            self.rval, self.frame = cam.read()
            print("cam is open")
        else:
            
            self.rval = False
            print("rval is False")
    
        while self.rval:
#            print("entering thread loop")
            self.rval, self.frame = cam.read()
#            print("rval is")
#            print(self.rval)
            if self.write == True: #output from measure()
                print("going to write")
                self.savelocation = path + str(self.camID) + str(self.j) + picname
                cv2.imwrite(self.savelocation, self.frame)
                self.j += 1
                
                print("camID = " + str(self.camID) + "\n j = " + str(self.j))
                self.write = False #to return to top of loop, reading frames
#                detect_humans(self.savelocation,self.camID)
            if self.end:  # to kill the thread input self.kill = True
                cam.release()
                break
#            if exit_event.is_set():
#                print("cam released")
#                cam.release()
#                break
            
    def savepic(self):
        cv2.imwrite(path + picname, self.frame)
        self.cam.release()
        print("cam is released")
        
    def kill(self):
        self.end = True
#   h
        
#def detect_humans(path,camID):
#    img = PILImage.open(path) #open image using PIL class 
#    print(type(img))
#    d = Detector(gpu_id=0)
#    img_arr = np.array(img.resize((d.network_width(), d.network_height())))
#    detections = d.perform_detect(image_path_or_buf=img_arr, show_image=False)
#    peoplenumber = 0
#    maxconfi = []
#    for detection in detections:
#        box = detection.left_x, detection.top_y, detection.width, detection.height
#        print(f'{detection.class_name.ljust(10)} | {detection.class_confidence * 100:.1f} % | {box}') 
#        if detection.class_name == 'person'and detection.class_confidence * 100 >= 50:
#            peoplenumber += 1
#            maxconfi.append(int(detection.class_confidence*100))
#    if peoplenumber >= 1:
#        gotpeople = True
#        detectconfi = max(maxconfi)
#        print(str(detectconfi))
#
#    else:
#        gotpeople = False
#        detectconfi = 0  # cannot be empty 
#    print('got {} people'.format(peoplenumber))   
#    img.close()
#    return gotpeople, peoplenumber, detectconfi
    
#    def write(self):
#        cv2.imwrite(path + "camera.jpg")
#        cam.release()
#        print("release via function")
#        cv2.destroyAllwindows()

##setting signaling 
#exit_event = threading.Event()
#def signal_handler(sigum, frame):
#    exit_event.set()
#
#signal.signal(signal.SIGINT, signal_handler)

# Create threads as follows
thread1 = camThread("Camera 1", 0)
thread2 = camThread("Camera 2", 2)
# thread3 = camThread("Camera 3", 2)



thread1.start()
print("thread is alive " + str(thread1.is_alive()))
thread2.start()
# thread3.start()

#thread1.daemon = True


#thread1.join()
print()
print("Active threads", threading.activeCount)

threadtuple = (thread1, thread2)
#for i in threadtuple:
#    i.end = True

def measure():
  # This function measures a distance
  # Pulse the trigger/echo line to initiate a measurement
    GPIO.output(GPIO_TRIGECHO, 1)
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGECHO, 0)
  #ensure start time is set in case of very quick return
    start = time.time()
    print("start")
    
# set line to input to check for start of echo response
    GPIO.setup(GPIO_TRIGECHO, GPIO.IN)
    while GPIO.input(GPIO_TRIGECHO)==0:
        start = time.time()

  # Wait for end of echo response
    while GPIO.input(GPIO_TRIGECHO)==1:
        stop = time.time()
#        print('stop here')
  
    GPIO.setup(GPIO_TRIGECHO, GPIO.OUT)
    GPIO.output(GPIO_TRIGECHO, 0)

    elapsed = stop-start
    distance = (elapsed * 34300)/2.0
    time.sleep(0.1)
    return distance

def detect_crossing():
    old_dist = 1
    i = 0
    while True:
        print("entering True loop")
        print("old distance: {}".format(old_dist))
        dist = measure()
        i += 1
        print("distance: ",dist,"/n iteration: ", i)
        dist_diff = old_dist - dist
        print("dist difference: {}".format(dist_diff))
        if dist_diff > 30:
            return True

#while True:
#    print("main Loop")
#    thread1.write = fakemeasure()
#    thread2.write = thread1.write
#    time.sleep(1)
#    for i in threadtuple:
#        print(i.savelocation)
#        gotpeople, peoplenumber, maxconfi= detect_humans(i.savelocation, i.camID)

class MinimalPublisher(Node):
      def __init__(self):
         super().__init__('minimal_publisher')
         self.publisher_ = self.create_publisher(SupportImage, 'SupportImage', 10)
         timer_period = 0.5  # seconds
         self.timer = self.create_timer(timer_period, self.timer_callback)
#         self.robot_number = input("please enter support robot number:\n")
         self.i = 0
         self.img_list = list()
         self.side = ("L","R")
         self.j = 0
#         self.im_list = []
#         self.cv_image = cv2.imread(path) ### an RGB image 
#         if self.cv_image.all == None:
#             print("no img")
#         cv2.imshow("pub",self.cv_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
         self.bridge = CvBridge()

      def timer_callback(self):
          try: 
              
              thread1.write = detect_crossing()
              thread2.write = thread1.write
              time.sleep(1)
              msg = SupportImage()
              for i in threadtuple:
                  print(i.savelocation)
                  cv_image = i.frame
#                  cv_image = cv2.imread(i.savelocation)
                  img = self.bridge.cv2_to_imgmsg(np.array(cv_image),"bgr8")
                  self.img_list.append(img)
#                  self.j +=1
                  
              msg.image1.img = self.img_list[0]
              msg.image1.side = self.side[0]
              msg.image2.img = self.img_list[1]
              msg.image2.side = self.side[1]
              msg.robotid = robot_id
              msg.time = int(time.time())
              print("unix epoch: " , str(msg.time))
              self.publisher_.publish(msg)
              self.get_logger().info('Publishing an image' + str(self.i))
              self.i +=1
#              print("callback started {} camID {}".format(self.i, str(i.camID)))
              
          except KeyboardInterrupt:
              for thread in threadtuple:
                  thread.end = True
#      def timer_callback(self):
#          try: 
#              
#              thread1.write = fakemeasure()
#              thread2.write = thread1.write
#              time.sleep(1)
#              for i in threadtuple:
#                  print(i.savelocation)
#                  gotpeople, peoplenumber, maxconfi = detect_humans(i.savelocation, i.camID)
#                  msg = Image()
#                  cv_image = cv2.imread(i.savelocation)
#                  msg = self.bridge.cv2_to_imgmsg(np.array(cv_image),"bgr8")
#            #            msg.header.stamp = node.get_clock().now().to_msg()
#            #          msg.header = "robot 1"
#                  msg.header.frame_id = "this is support robot no. " + str(self.robot_number) + "\n callback no." + str(self.i)
#                  self.publisher_.publish(msg)
#                  self.get_logger().info('Publishing an image')
#                  print("callback started {} camID {}".format(self.i, str(i.camID)))
#                  
#              self.i+=1
#              
#          except KeyboardInterrupt:
#              for thread in threadtuple:
#                  thread.end = True
              
def main(args=None):
    rclpy.init(args=args)
    print("ready for pic")
#    take_pic()
    minimal_publisher = MinimalPublisher()
    rclpy.spin(minimal_publisher)
    print('spin done')
    minimal_publisher.destroy_node()

    rclpy.shutdown()
    print("shutdown")

if __name__ == '__main__':
    main()
