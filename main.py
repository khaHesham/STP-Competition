"""
Example code using the judge module
"""
import time

# pylint: disable=import-error
import cv2
import keyboard
import numpy as np
import math

from machathon_judge import Simulator, Judge



def lane_img(img):
    lower_red = np.array([40, 40, 150])
    upper_red = np.array([70, 70, 190])
    mask = cv2.inRange(img, lower_red, upper_red)
    result = cv2.bitwise_and(img, img, mask = mask)
    return result

def line_detection(img, mn, mx):
    # Convert the img to grayscale
    gray = cv2.cvtColor(img[mn:mx, :], cv2.COLOR_BGR2GRAY)
  
    # Apply edge detection method on the image
    edges_r = cv2.Canny(gray[:, 320:], 50, 150)
    edges_l = cv2.Canny(gray[:, :320], 50, 150)
  
    # This returns an array of r and theta values
    lines_r = cv2.HoughLines(edges_r, 1, np.pi/180, 40)
    lines_l = cv2.HoughLines(edges_l, 1, np.pi/180, 40)
    if np.any(lines_r):
        slopes_r = 1/np.tan(lines_r[:, 0, 1])
        
        avg_slope_r = sum(slopes_r)/len(slopes_r)
        avg_slope_r = np.clip(avg_slope_r, -15, 15)

    else:
        avg_slope_r = 0

    if np.any(lines_l):
        slopes_l = 1/np.tan(lines_l[:, 0, 1])

        avg_slope_l = sum(slopes_l)/len(slopes_l)
        avg_slope_l = np.clip(avg_slope_l, -15, 15)
    else:
        avg_slope_l = 0

    if avg_slope_l * avg_slope_r < 0:
        avg_slope = 0
    else:
        avg_slope = (avg_slope_r + avg_slope_l)/2

    
    #avg_slope = np.clip(avg_slope, -15, 15)
    return avg_slope

def steer(slope, factor):
    # -ve left
    # +ve right
    # 4 is max
    slope = -factor * np.pi/180 * slope
    
    return slope


class counter:
    def __init__(self):
        self.i = 0
        pass







def run_car(simulator: Simulator) -> None:
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
    #fps_counter.step()

    #_, vel = simulator.get_state()

    # Get the image and show it
    img = simulator.get_image()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mod_img = lane_img(img)

    shift = 50 # 30 with 50, 
    turning_close = line_detection(img, 380, 420)
    turning_wide = line_detection(img, 290, 320)



    if abs(turning_close) < 0.0001 and abs(turning_wide) < 0.0001:
        simulator.set_car_velocity(11)
        print("FASTEST")

    elif abs(turning_close) < 0.0001 and abs(turning_wide) >= 0.0001:
        simulator.set_car_velocity(9)
        print("Normal")

    elif abs(turning_close) >= 0.0001 and abs(turning_wide) >= 0.0001:
        simulator.set_car_velocity(5)
        print("slow")
    else:
        simulator.set_car_velocity(5)
        print("slow")

    turning_close = steer(turning_close, 50)
    simulator.set_car_steering(turning_close)


    
    
    cv2.imshow(f"image", img)
    cv2.waitKey(1)

    


if __name__ == "__main__":
    i = counter()

    judge = Judge(team_code="T****", zip_file_path="main.zip")

    # Pass the function that contains your main solution to the judge
    judge.set_run_hook(run_car)

    # Start the judge and simulation
    judge.run(send_score=False, verbose=True)
