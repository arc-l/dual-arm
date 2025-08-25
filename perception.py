import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from ur_tools.camera.realsense_camera import Camera 
from ur_tools.ur_control.ur5_VGA10 import UR5_VGA10 

class Perception:
    def __init__(self, camera:Camera|None=None, use_letters:bool=True):
        # Initialize camera or any other sensors here
        if camera is None:
            self.camera = Camera(calibrated=True, on_hand=False)
        else:
            self.camera = camera
        self.aruco_dict_objs = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.aruco_detector_objs = cv2.aruco.ArucoDetector(self.aruco_dict_objs)
        self.use_letters = use_letters

    def get_data(self):
        return self.camera.get_data()
    
    def show_data(self):
        self.camera.show_data()
    
    def stream(self):
        self.camera.show_video_realtime()
    
    def update_cube_poses(self, visualize=False):
        '''
        update seen object poses
        '''
        color_img, depth_img = self.get_data()
        
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = self.aruco_detector_objs.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            print("No ArUco markers detected.")
            return None
        
        if visualize:
            plt.figure(figsize=(12, 8))
            plt.imshow(color_img)
            plt.title('Detected ArUco Markers and Corners')
            plt.axis('off')
            # Draw corners and markers
            for i, marker_id in enumerate(ids.flatten()):
                marker_corners = corners[i][0]
                
                # Draw the corner points
                for j, corner in enumerate(marker_corners):
                    plt.plot(corner[0], corner[1], 'ro', markersize=8, alpha=0.8)
                    # Label each corner
                    plt.annotate(f'C{j}', (corner[0], corner[1]), 
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=8, color='red', weight='bold')
                
                # Draw lines connecting the corners to form the marker boundary
                marker_corners_closed = np.vstack([marker_corners, marker_corners[0]])
                plt.plot(marker_corners_closed[:, 0], marker_corners_closed[:, 1], 
                        'g-', linewidth=2, alpha=0.8)
                
                # Draw marker center and ID
                marker_center = np.mean(marker_corners, axis=0)
                plt.plot(marker_center[0], marker_center[1], 'bo', markersize=10)
                plt.annotate(f'ID: {marker_id}', marker_center, 
                        xytext=(10, -10), textcoords='offset points',
                        fontsize=10, color='blue', weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            plt.tight_layout()
            plt.show()
                
        tag_loc_camera = {}
        for i, marker_id in enumerate(ids.flatten()):
            marker_corners = corners[i][0]
            marker_center_2d = np.mean(marker_corners, axis=0)
            tag_loc_camera[marker_id] = marker_center_2d

        world_pos_to_ee = {}
        for key, cam_pt in tag_loc_camera.items():
            x, y = cam_pt
            z = depth_img[int(y), int(x)]
            x = (x - self.camera.intrinsics[0, 2]) * z / self.camera.intrinsics[0, 0]
            y = (y - self.camera.intrinsics[1, 2]) * z / self.camera.intrinsics[1, 1]
            world_pos_to_ee[int(key)] = self.camera.pos_in_camera_to_ee(np.array([[x, y,z]]))[0]
        return world_pos_to_ee
    

if __name__ == "__main__":
    p = Perception()
    p.stream()
    p.update_cube_poses(visualize=True)
    