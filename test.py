"""
Example code using the judge module
"""
import time

# pylint: disable=import-error
import cv2
import keyboard
import laneDetector as ld
# import yarab as yarab
import math
import sys
import numpy as np

from machathon_judge import Simulator, Judge


class FPSCounter:
    def __init__(self):
        self.frames = []
        self.angle=0
        self.c=0
        self.thrsh=0.5

    def step(self):
        self.frames.append(time.monotonic())

    def get_fps(self):
        n_seconds = 5

        count = 0
        cur_time = time.monotonic()
        for f in self.frames:
            if cur_time - f < n_seconds:  # Count frames in the past n_seconds
                count += 1

        return count / n_seconds
    
#############################################################################################################
#############################################################################################################
#############################################################################################################

def segmented(bgr_image):
    lower_rgb=np.array([50,50,165],dtype="uint8")
    upper_rgb=np.array([70,70,190],dtype="uint8")    
    skin_region=cv2.inRange(bgr_image,lower_rgb,upper_rgb)
    bgr=cv2.erode(skin_region,np.ones((5,5), np.uint8))
    cv2.imshow("segmented",bgr)
    
    return bgr

def detect_edges(frame):
    # filter for blue lane lines
    # hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # #cv2.imshow("HSV",hsv)
    # lower_blue = np.array([90, 120, 0], dtype = "uint8")
    # upper_blue = np.array([150, 255, 255], dtype="uint8")
    # mask = cv2.inRange(hsv,lower_blue,upper_blue)
    # #cv2.imshow("mask",mask)
    
    # detect edges
    # edges = cv2.Canny(mask, 50, 100)
    edges=segmented(frame)
    #cv2.imshow("edges",edges)
    
    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus lower half of the screen
    polygon = np.array([[
        (0, height),
        (0,  height/2),
        (width , height/2),
        (width , height),
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    
    cropped_edges = cv2.bitwise_and(edges, mask)
    #cv2.imshow("roi",cropped_edges)
    
    return cropped_edges

def detect_line_segments(cropped_edges):
    rho = 1  
    theta = np.pi / 180  
    min_threshold = 10  
    
    line_segments = cv2.HoughLinesP(cropped_edges, rho, theta, min_threshold, 
                                    np.array([]), minLineLength=5, maxLineGap=150)

    return line_segments


def average_slope_intercept(frame, line_segments):
    lane_lines = []
    
    if line_segments is None:
        print("no line segments detected")
        return lane_lines

    height, width,_ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary
    
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                print("skipping vertical lines (slope = infinity")
                continue
            
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    return lane_lines

def make_points(frame, line):
    height, width, _ = frame.shape
    
    slope, intercept = line
    
    y1 = height  # bottom of the frame
    y2 = int(y1 / 2)  # make points from middle of the frame down
    
    if slope == 0:
        slope = 0.1
        
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    return [[x1, y1, x2, y2]]

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=6):
    line_image = np.zeros_like(frame)
    
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
                
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    
    return line_image


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    
    steering_angle_radian = steering_angle / 180.0 * math.pi
    
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)
    
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    
    return heading_image

def get_steering_angle(frame, lane_lines):
    
    height,width,_ = frame.shape
    
    if len(lane_lines) == 2:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        mid = int(width / 2)
        x_offset = (left_x2 + right_x2) / 2 - mid
        y_offset = int(height / 2)
        
    elif len(lane_lines) == 1:
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
        y_offset = int(height / 2)
        
    elif len(lane_lines) == 0:
        x_offset = 0
        y_offset = int(height / 2)
        
    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  
    steering_angle = angle_to_mid_deg + 90
    
    return steering_angle

lastTime = 0
lastError = 0

#proportional and derivative constants
kp = 0.05
kd = kp * 0.65

#variables used for stop sign timing
i=0
j=0
frame_count = 0
frame_num = -20

#initial speed
throttle_speed = 1
counter=0

#lists to be extracted to make graphs
frames = []
errors = []
steering_pwms = []
throttle_pwms = []
p_response = []
d_response = []

def Pilot(frame):

    global lastTime
    global lastError 

    #proportional and derivative constants
    global kp 
    global kd 

    #variables used for stop sign timing
    global i
    global j
    global frame_count 
    global frame_num 

    #initial speed
    global throttle_speed 

    #lists to be extracted to make graphs
    global frames
    global errors
    global steering_pwms
    global throttle_pwms 
    global p_response 
    global d_response 
    # ret,frame = video.read()
    # i += 1


    # cv2.imshow("image", img)  # 640 * 480
    frames.append(i)

    # frame_count += 1
    
    #set camera
    #cv2.imshow("original",frame)
    edges = detect_edges(frame)
    roi = region_of_interest(edges)
    line_segments = detect_line_segments(roi)
    lane_lines = average_slope_intercept(frame,line_segments)
    lane_lines_image = display_lines(frame,lane_lines)
    steering_angle = get_steering_angle(frame, lane_lines)
    heading_image = display_heading_line(lane_lines_image,steering_angle)
    # print(steering_angle)
    cv2.imshow("heading line",heading_image)
    return steering_angle,heading_image
    

#############################################################################################################
#############################################################################################################
#############################################################################################################



def run_car(simulator: Simulator) -> None:   # 170,55,55
    """
    Function to control the car using keyboard

    Parameters
    ----------
    simulator : Simulator
        The simulator object to control the car
        The only functions that should be used are:
        - get_image()
        - set_car_steering()
        - set_car_velocity()
        - get_state()
    """
    global lastTime
    global lastError 

    #proportional and derivative constants
    global kp 
    global kd 

    #initial speed
    global throttle_speed 

    #lists to be extracted to make graphs
    global frames
    global errors
    global steering_pwms
    global throttle_pwms 
    global p_response 
    global d_response 

    global counter


    fps_counter.step()

    # Get the image and show it
    img = simulator.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fps = fps_counter.get_fps()

    # draw fps on image
    cv2.putText(
        img,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("image", img)  # 640 * 480
    cv2.waitKey(1)

    # API = ld.DetectorAPI()

    # Testing our Interface with API
    # API.Run()  [dont run unless u want to see the same video i uploaded on whatsapp]
    # API.Detect(img)
    # yarab.yarab(img)
    steering_angle,heading_image = Pilot(img)

    


    # Control the car using keyboard
    # steering = 0
    # if keyboard.is_pressed("a"):
    #     steering = 1
    # elif keyboard.is_pressed("d"):
    #     steering = -1

    # throttle = 0
    # if keyboard.is_pressed("w"):
    #     throttle = 1
    # elif keyboard.is_pressed("s"):
    #     throttle = -1

    # simulator.set_car_steering(steering * simulator.max_steer_angle / 1.7)
    # simulator.set_car_velocity(throttle * 25)

    # #set throttle
    simulator.set_car_velocity(2)

    #PD controller
    counter = counter + 1

    now = time.time()
    dt = now - lastTime

    deviation = steering_angle - 90
    error = abs(deviation)
    errors.append(error)
    steering = 0
    
    derivative = kd * (error - lastError) / dt
    proportional = kp * error
    PD = int(steering + derivative + proportional)
    steering = abs(PD)
    # print("steering: " + str(steering))   
    p_response.append(proportional)
    d_response.append(derivative) 


    print("steering angle  " + str(steering_angle), "steering  " + str(steering))

    currAngle ,currSpeed = simulator.get_state()
    # print("curr angle: "+str(currAngle))
    # print("curr speed: "+str(currSpeed))

        

    #steering logic using PD values   -ve -> right    left <- +ve
    new_val = 7.5

    if counter%5==0:
        if deviation < 15 and deviation > -15:
            print("Not Steering")
            # print("Deviation: " + str(deviation))
            new_val = 7.5
            deviation = 0
            error = 0
            # simulator.set_car_steering(0)         # No Steering
            simulator.set_car_velocity(currSpeed + 0.5)
                

        elif deviation > 50:
            print("Steering Right")
            # print("Deviation: " + str(deviation))
            new_val = 5-steering * 5.5
            # print("right val: " + str(new_val))
            simulator.set_car_velocity(currSpeed - 0.5)
            simulator.set_car_steering(new_val*np.pi/180)  
            
        elif deviation > 45:
            print("Steering Right")
            # print("Deviation: " + str(deviation))
            new_val = 5-steering * 3.0
            # print("right val: " + str(new_val))
            simulator.set_car_velocity(currSpeed - 0.5)
            simulator.set_car_steering(new_val*np.pi/180)  

        elif deviation > 40:
            print("Steering Right")
            # print("Deviation: " + str(deviation))
            new_val = 5-steering * 2.5
            # print("right val: " + str(new_val))
            simulator.set_car_velocity(currSpeed - 0.5)
            simulator.set_car_steering(new_val*np.pi/180)  

        elif deviation > 35:
            print("Steering Right")
            # print("Deviation: " + str(deviation))
            new_val = 5-steering * 1.5
            # print("right val: " + str(new_val))
            simulator.set_car_velocity(currSpeed - 0.5)
            simulator.set_car_steering(new_val*np.pi/180) 

        elif deviation > 30:
            print("Steering Right")
            # print("Deviation: " + str(deviation))
            new_val =  5-steering * 1.0
            # print("right val: " + str(new_val))
            simulator.set_car_velocity(currSpeed - 0.5)
            simulator.set_car_steering(new_val*np.pi/180)

        elif deviation > 25:
            print("Steering Right")
            # print("Deviation: " + str(deviation))
            new_val = 5-steering * 0.6
            # print("right val: " + str(new_val))
            simulator.set_car_velocity(currSpeed -0.5)
            simulator.set_car_steering(new_val*np.pi/180) 

        elif deviation > 20:
            print("Steering Right")
            # print("Deviation: " + str(deviation))
            new_val = 5-steering * 0.5
            # print("right val: " + str(new_val))
            simulator.set_car_velocity(2)
            simulator.set_car_steering(new_val*np.pi/180)  

        elif deviation > 15:
            print("Steering Right")
            # print("Deviation: " + str(deviation))
            new_val = 5-steering * 0.1
            # print("right val: " + str(new_val))
            simulator.set_car_velocity(currSpeed + .5)
            simulator.set_car_steering(new_val*np.pi/180) 

        elif deviation > 10:
            print("Steering right")
            # print("Deviation: " + str(deviation))
            new_val = 5-steering * 0.05
            # print("right val: " + str(new_val))
            simulator.set_car_velocity(currSpeed + 1)
            simulator.set_car_steering(new_val*np.pi/180 )  
            
        
        elif deviation < -50:
            print("Steering Left")
            # print("Deviation: " + str(deviation))
            new_val = steering * 5.5
            # print("left val: " + str(new_val))
            throttle_speed = 3
            #   simulator.set_car_velocity(throttle_speed)
            simulator.set_car_velocity(currSpeed - 0.5)
            simulator.set_car_steering(new_val*np.pi/180)  

        elif deviation < -45:
            print("Steering Left")
            # print("Deviation: " + str(deviation))
            new_val =  5+steering * 3
            # print("left val: " + str(new_val))
            throttle_speed = 3
            #   simulator.set_car_velocity(throttle_speed)
            simulator.set_car_velocity(currSpeed - 0.5)
            simulator.set_car_steering(new_val*np.pi/180)

        elif deviation < -40:
            print("Steering Left")
            # print("Deviation: " + str(deviation))
            new_val = 5+steering * 2.5
            # print("left val: " + str(new_val))
            throttle_speed = 3
            #   simulator.set_car_velocity(throttle_speed)
            simulator.set_car_velocity(currSpeed - 0.5)
            simulator.set_car_steering(new_val*np.pi/180)

        elif deviation < -35:
            print("Steering Left")
            # print("Deviation: " + str(deviation))
            new_val = 5+steering * 1.5
            # print("left val: " + str(new_val))
            throttle_speed = 3
            #   simulator.set_car_velocity(throttle_speed)
            simulator.set_car_velocity(currSpeed - 0.5)
            simulator.set_car_steering(new_val*np.pi/180)

        elif deviation < -30:
            print("Steering Left")
            # print("Deviation: " + str(deviation))
            new_val =  5+steering * 1
            # print("left val: " + str(new_val))
            throttle_speed = 3
            #   simulator.set_car_velocity(throttle_speed)
            simulator.set_car_velocity(currSpeed - 0.5)
            simulator.set_car_steering(new_val*np.pi/180)

        elif deviation < -25:
            print("Steering Left")
            # print("Deviation: " + str(deviation))
            new_val = 5+steering * 0.5
            # print("left val: " + str(new_val))
            throttle_speed = 3
            #   simulator.set_car_velocity(throttle_speed)
            simulator.set_car_velocity(currSpeed + 1)
            simulator.set_car_steering(new_val*np.pi/180)
        
        elif deviation < -20:
            print("Steering Left")
            # print("Deviation: " + str(deviation))
            new_val =  5+steering * 0.25
            # print("left val: " + str(new_val))
            simulator.set_car_velocity(currSpeed + 1)
            simulator.set_car_steering(new_val*np.pi/180 ) 

        elif deviation < -15:
            print("Steering Left")
            # print("Deviation: " + str(deviation))
            new_val = 5+steering *  0.01
            # print("left val: " + str(new_val))
            simulator.set_car_velocity(currSpeed + 1)
            simulator.set_car_steering(new_val*np.pi/180 ) 

        elif deviation < -10:
            print("Steering Left")
            # print("Deviation: " + str(deviation))
            new_val = 5+steering * 0.005
            # print("left val: " + str(new_val))
            simulator.set_car_velocity(currSpeed + 1)
            simulator.set_car_steering(new_val*np.pi/180)  

      
    # #append lists
    # steering_pwms.append(new_val)
    # throttle_pwms.append(throttle_speed)

        lastError = error
        lastTime = time.time()



angle = 0
c = 1
thrsh=1
i = 0
def AutoPilot(simulator: Simulator) -> None:
    """
    lets win the Game ;) [inshalah]

    Parameters
    ----------
    simulator : Simulator
        The simulator object to control the car
        The only functions that should be used are:
        - get_image()
        - set_car_steering()
        - set_car_velocity()
        - get_state()
    """

    global angle
    global c
    global thrsh
    global i

    fps_counter.step()

    # Get the image and show it
    img = simulator.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    fps = fps_counter.get_fps()

    # draw fps on image
    cv2.putText(
        img,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("image", img)
    cv2.waitKey(1)

    # Initiate our API Object
    API = ld.DetectorAPI()

    # Testing our Interface with API
    # API.Run()  [dont run unless u want to see the same video i uploaded on whatsapp]

    old_angle= angle

    frame,img_b,img_w,final,Angle=API.Detect(img)

    cv2.imshow('front_view', frame)
    cv2.imshow('ROI', img_b)
    cv2.imshow('Sky_view', img_w)
    cv2.imshow('final', final)

    angle = Angle

    i=i+1

    
    """
                                Controlling Our Car Begins Here
    side Notes:
    -----------
    [#] you can make any getter function u want for any variable in the API 
    just make sure to initiate it in __init__ function self.something = 0 (as example)

    [#] i guess we will need steering angle and radius only from DetectorApi. (we will see)

    [#] the initial condition of the car is not always in the middle 
    we need to consider centering the car in our solution.

    [#] this is our initial code we will change it a lot till the deadline just take
    a look at the DetectorAPI to understand how it works and feel free to ask me anything about it.
 
    """
    simulator.set_car_velocity(3)

    currAngle ,currSpeed = simulator.get_state()

    if(i % 5==0 and abs(angle - old_angle) < thrsh):
        # c= c + 1
        simulator.set_car_velocity(3) 
    else:
        # if currSpeed  > 2:
            # simulator.set_car_velocity(currSpeed-1)
        simulator.set_car_velocity(0)
        simulator.set_car_steering(currAngle - 0.009*3.14*angle/180)

        
    

     

if __name__ == "__main__":
    # Initialize any variables needed
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    fps_counter = FPSCounter()

    # You should modify the value of the parameters to the judge constructor
    # according to your team's info
    judge = Judge(team_code="your_new_team_code", zip_file_path="your_solution.zip")

    # Pass the function that contains your main solution to the judge

    judge.set_run_hook(run_car)

    # Start the judge and simulation
    judge.run(send_score=False, verbose=True)
