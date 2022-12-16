import cv2
#import habitat
import numpy as np
#from habitat.sims.habitat_simulator.actions import HabitatSimActions
import time

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
FINISH="f"
SAVE="s"

class ObstacleAvoidAgent():
    '''
    Agent to take the depth image to generate steering yaw rate to avoid 
    the obstacles surrounding it.
    TODO:
    - [] Gate detection.
    '''
    def __init__(self):
        self.lambda_avoid = 10 #
        self.lambda_goto = 2 #
        self.constant_obst = 0.5 #
        self.num_of_bins = 8 # 12
        self.sigma = 12 # 
        self.image = None
        self.strip = None
        self.depth_set = np.zeros(self.num_of_bins)
    
    
    def image_read_from_png(self, image):
        self.image = (image / 255).astype(np.float32)


    def image_read(self, image):
        self.image = image.astype(np.float32)


    def _get_horizontal_strip(self, offset):
        '''
        Take the depth image as input, generate a strip that has a
        number of bins around the middle of the depth image.
        offset: offset due to pitch angle. Low if pitch angle is small.
        For habitat depth image: 324x256
        Return: a strip cut from the original depth image
        '''
        assert self.image is not None
        self.bin_width = len(self.image[0]) // self.num_of_bins # 40/8=5
        self.horiz_middle = len(self.image) // 2
        bin_upper = self.horiz_middle - 2 * self.bin_width
        bin_lower = self.horiz_middle + 2 * self.bin_width + 4

        self.strip = self.image[(bin_upper + offset) : (bin_lower + offset)]


    def _generate_depth_set(self):
        '''
        Generate a depth set along the strip.
        Return: a numpy.array in which each element_i is the averaged depth in that bin_i.
        '''
        assert self.strip is not None
        averaged_depth = self.strip.mean(axis=0)
        for i in range(self.num_of_bins):
            self.depth_set[i] = averaged_depth[i * self.bin_width : (i + 1) * self.bin_width].mean()
        #print("depth bin:", self.depth_set)
    
    
    def steering(self, pitch_offset):
        '''
        Based on the behavior arbitration scheme & depth bins, generate steering policy.
        Turning right is positive yaw.
        '''
        state = 'avoidance'
        self._get_horizontal_strip(offset=pitch_offset)
        self._generate_depth_set()
        relative_index = self.num_of_bins / 2 - 0.5

        # if obstacle in front
        edge_bin_indexes = [0, 1, 2, 5, 6, 7]
        edge_bins = [self.depth_set[i] for i in edge_bin_indexes]
        center_bins = [self.depth_set[3], self.depth_set[4]]
        if np.average(center_bins) >= 1.1*(np.average(edge_bins)):
            print("obstacle in front, turning")
            steer = -0.5
            state = 'obstacle_in_front'

        # if wall in front
        elif np.var(self.depth_set) <= 1e-5:
            #print("wall in front, turning")
            steer = -0.4
            state = 'wall_in_front'
        else:
            # behavior scheme
            steer = self.lambda_avoid * np.array([(relative_index - i) * np.exp(-self.constant_obst * self.depth_set[i]) * \
                        np.exp(-(i - relative_index)**2 / (2 * self.sigma**2)) for i in range(self.num_of_bins)]).sum()

        return -steer, state


    def steering_behavior(self, pitch_offset):
        '''
        Based on the behavior arbitration scheme & depth bins, generate steering policy.
        Turning right is positive yaw.
        '''
        state = 'avoidance'
        self._get_horizontal_strip(offset=pitch_offset)
        self._generate_depth_set()
        relative_index = self.num_of_bins / 2 - 0.5

        min_dips = np.argmin(self.depth_set)
        goal_steer = 0.3 * (relative_index - min_dips) # previously 0.2

        # if obstacle in front
        edge_bin_indexes = [0, 1, 2, 5, 6, 7]
        edge_bins = [self.depth_set[i] for i in edge_bin_indexes]
        center_bins = [self.depth_set[3], self.depth_set[4]]
        if np.average(center_bins) >= 1.09*(np.average(edge_bins)):
            #print("obstacle in front, turning")
            state = 'obstacle_in_front'
            steer = -1.0

        # if wall in front
        elif np.var(self.depth_set) <= 1e-5:
            #print("wall in front, turning")
            state = 'wall_in_front'
            steer = -0.5
        else:
            # behavior scheme
            steer = self.lambda_avoid * np.array([(relative_index - i) * np.exp(-self.constant_obst * self.depth_set[i]) * \
                        np.exp(-(i - relative_index)**2 / (2 * self.sigma**2)) for i in range(self.num_of_bins)]).sum()
            steer += goal_steer
        #print(steer, goal_steer, (steer + goal_steer))

        return -steer, state


    # def steering_visual(self, pitch_offset):
    #     '''
    #     Based on the behavior arbitration scheme & depth bins, generate steering policy.
    #     Turning right is positive yaw.
    #     '''
    #     state = 'avoidance'
    #     self._get_horizontal_strip(offset=pitch_offset)
    #     self._generate_depth_set()
    #     relative_index = self.num_of_bins / 2 - 0.5

    #     # if obstacle in front
    #     min_dips = np.argmin(self.depth_set)
    #     steer = (relative_index - min_dips)
    #     return -steer, state


    # def steering_zoo(self, pitch_offset):
    #     '''
    #     Based on the behavior arbitration scheme & depth bins, generate steering policy.
    #     Turning right is positive yaw.
    #     '''
    #     self._get_horizontal_strip(offset=pitch_offset)
    #     self._generate_depth_set()
    #     relative_index = self.num_of_bins / 2 - 0.5
    #     steer = self.lambda_avoid * np.array([(relative_index - i) * np.exp(-self.constant_obst * self.depth_set[i]) * \
    #                                 np.exp(-(i - relative_index)**2 / (2 * self.sigma**2)) for i in range(self.num_of_bins)]).sum()

    #     return -steer


# def depth_navigation():
#     env = habitat.Env(config=habitat.get_config("configs/tasks/pointnav_gibson.yaml"))

#     print("Environment creation successful")
#     observations = env.reset()
#     print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
#         observations["pointgoal_with_gps_compass"][0],
#         observations["pointgoal_with_gps_compass"][1]))
#     cv2.imshow("DEPTH", observations["depth"])
#     cv2.imshow("RGB", observations["rgb"])

#     # agent
#     agent = ObstacleAvoidAgent()

#     print("Agent stepping around inside environment.")

#     count_steps = 0
#     while not env.episode_over:
#         keystroke = cv2.waitKey(0)

#         if keystroke == ord(FORWARD_KEY):
#             action = HabitatSimActions.MOVE_FORWARD
#             #print("action: FORWARD")
#         elif keystroke == ord(LEFT_KEY):
#             action = HabitatSimActions.TURN_LEFT
#             #print("action: LEFT")
#         elif keystroke == ord(RIGHT_KEY):
#             action = HabitatSimActions.TURN_RIGHT
#             #print("action: RIGHT")
#         elif keystroke == ord(FINISH):
#             action = HabitatSimActions.STOP
#             #print("action: FINISH")
#         elif keystroke == ord(SAVE):
#             cv2.imwrite("./examples/images/depth_image_{}.png".format(count_steps),
#                         255 * observations["depth"])
#             # print(observations["depth"].shape)
#             # print(observations["depth"].dtype)
#         else:
#             print("INVALID KEY")
#             continue

#         agent.image_read(observations["depth"])
#         steer = agent.steering(pitch_offset=0)
#         print("steering: {}".format('right' if steer > 0 else 'left'), steer)
#         observations = env.step(action)
#         count_steps += 1

#         # print("Destination, distance: {:3f}, theta(radians): {:.2f}".format(
#         #     observations["pointgoal_with_gps_compass"][0],
#         #     observations["pointgoal_with_gps_compass"][1]))
        
#         cv2.imshow("DEPTH", observations["depth"])
#         cv2.imshow("RGB", observations["rgb"])

#     print("Episode finished after {} steps.".format(count_steps))

#     if (
#         action == HabitatSimActions.STOP
#         and observations["pointgoal_with_gps_compass"][0] < 0.2
#     ):
#         print("you successfully navigated to destination point")
#     else:
#         print("your navigation was unsuccessful")


# if __name__ == "__main__":
#     depth_navigation()