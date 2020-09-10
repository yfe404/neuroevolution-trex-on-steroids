from skimage.transform import resize
import numpy as np

class Preprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def process(self, frame):
        roi_height, roi_width = frame.shape[0], int(frame.shape[1] * .68)
        processed = np.zeros((roi_height, roi_width))

        roi = frame[:, :roi_width, 0]
        all_obstacles_idx = roi > 50
        processed[all_obstacles_idx] = 1
        unharmful_obstacles_idx = roi > 200
        processed[unharmful_obstacles_idx] = 0

        processed = resize(processed, (self.height, self.width, 1))
        processed = processed / 255.0
        return processed

    def get_initial_state(self, first_frame):
        self.state = np.array([first_frame, first_frame, first_frame, first_frame])
        return self.state

    def get_updated_state(self, next_frame):
        self.state =  np.array([*self.state[-3:], next_frame])
        return self.state


class SimplePreprocessor:
    def __init__(self):
        pass

    def process(self, frame):
        return frame

    def get_initial_state(self, first_frame):
        return self.get_updated_state(first_frame)

    def get_updated_state(self, next_frame):
#        print(next_frame)
        speed = next_frame.get("speed") 
        x_pos = next_frame.get("xPos") 
        y_pos = next_frame.get("yPos") 
        has_obstacles = len(next_frame.get("obstacles")) > 0

        next_obstacle_x_pos = -1
        next_obstacle_y_pos = -1
        type_obstacle = -1
        if has_obstacles:
            next_obstacle_x_pos = next_frame.get("obstacles")[0].get('xPos')
            next_obstacle_y_pos = next_frame.get("obstacles")[0].get('yPos')
            
            if next_frame.get("obstacles")[0].get('typeConfig').get('type') == 'CACTUS_SMALL':
                type_obstacle = 0.33
            if next_frame.get("obstacles")[0].get('typeConfig').get('type') == 'CACTUS_LARGE':
                type_obstacle = 0.66
            if next_frame.get("obstacles")[0].get('typeConfig').get('type') == 'PTERODACTYL':
                type_obstacle = 1.0 

            assert type_obstacle > -1

            
        self.state = np.array([speed, x_pos, y_pos, next_obstacle_x_pos, next_obstacle_y_pos, type_obstacle])

#        print(self.state)
        return self.state

