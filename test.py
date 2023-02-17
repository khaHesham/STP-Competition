"""
Example code using the judge module
"""
import time

# pylint: disable=import-error
import cv2
import keyboard
import laneDetector as ld

from machathon_judge import Simulator, Judge


class FPSCounter:
    def __init__(self):
        self.frames = []

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

    API = ld.DetectorAPI()

    # Testing our Interface with API
    # API.Run()  [dont run unless u want to see the same video i uploaded on whatsapp]
    API.Detect(img)

    # Control the car using keyboard
    steering = 0
    if keyboard.is_pressed("a"):
        steering = 1
    elif keyboard.is_pressed("d"):
        steering = -1

    throttle = 0
    if keyboard.is_pressed("w"):
        throttle = 1
    elif keyboard.is_pressed("s"):
        throttle = -1

    simulator.set_car_steering(steering * simulator.max_steer_angle / 1.7)
    simulator.set_car_velocity(throttle * 25)


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
    API.Detect(img)

    
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
