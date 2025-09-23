import cv2
from constants import CELL_HEIGHT, CELL_WIDTH, KITCHEN_HEIGHT
from utils.conversions import convert_feet_to_pixel_distance, convert_pixel_distance_to_feet
import numpy as np

class CourtDiagram:

    def __init__(self, frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 450
        self.buffer = 50
        self.padding = 25

        self.set_background_position(frame)
        self.set_diagram_position()
        self.set_court_keypoints()
        self.set_court_lines()


    def set_background_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    
    def set_diagram_position(self):
        self.court_start_x = self.start_x + self.padding
        self.court_end_x = self.end_x - self.padding
        self.court_start_y = self.start_y + self.padding
        self.court_end_y = self.end_y - self.padding
        self.court_drawing_width = self.court_end_x - self.court_start_x
    
    def convert_feet_pixels(self, feet):
        return convert_feet_to_pixel_distance(feet, self.court_drawing_width, CELL_WIDTH * 2)

    def set_court_keypoints(self):
        self.drawing_kp = [0]*24

        # Point 1
        self.drawing_kp[0] = int(self.court_start_x)
        self.drawing_kp[1] = int(self.court_end_y)
        # Point 2
        self.drawing_kp[2] = int(self.court_start_x + self.convert_feet_pixels(CELL_WIDTH))
        self.drawing_kp[3] = int(self.court_end_y)
        # Point 3
        self.drawing_kp[4] = int(self.court_start_x + self.convert_feet_pixels(CELL_WIDTH * 2))
        self.drawing_kp[5] = int(self.court_end_y)
        # Point 4
        self.drawing_kp[6] = int(self.court_start_x + self.convert_feet_pixels(CELL_WIDTH * 2))
        self.drawing_kp[7] = int(self.court_end_y) - self.convert_feet_pixels(CELL_HEIGHT)
        # Point 5
        self.drawing_kp[8] = int(self.court_start_x)
        self.drawing_kp[9] = int(self.court_start_y)
        # Point 6
        self.drawing_kp[10] = int(self.court_start_x + self.convert_feet_pixels(CELL_WIDTH))
        self.drawing_kp[11] = int(self.court_start_y)
        # Point 7
        self.drawing_kp[12] = int(self.court_start_x + self.convert_feet_pixels(CELL_WIDTH * 2))
        self.drawing_kp[13] = int(self.court_start_y)
        # Point 8
        self.drawing_kp[14] = int(self.court_start_x + self.convert_feet_pixels(CELL_WIDTH * 2))
        self.drawing_kp[15] = int(self.court_start_y + self.convert_feet_pixels(CELL_HEIGHT))
        # Point 9
        self.drawing_kp[16] = int(self.court_start_x + self.convert_feet_pixels(CELL_WIDTH))
        self.drawing_kp[17] = int(self.court_end_y - self.convert_feet_pixels(CELL_HEIGHT))
        # Point 10
        self.drawing_kp[18] = int(self.court_start_x)
        self.drawing_kp[19] = int(self.court_end_y - self.convert_feet_pixels(CELL_HEIGHT))
        # Point 11
        self.drawing_kp[20] = int(self.court_start_x)
        self.drawing_kp[21] = int(self.court_start_y + self.convert_feet_pixels(CELL_HEIGHT))
        # Point 12
        self.drawing_kp[22] = int(self.court_start_x + self.convert_feet_pixels(CELL_WIDTH))
        self.drawing_kp[23] = int(self.court_start_y + self.convert_feet_pixels(CELL_HEIGHT))

    def set_court_lines(self):
        self.lines = [
            # Outer court
            (1, 5), 
            (3, 7),
            (5, 7),
            (1, 3),
            # Kitchen Lines
            (8, 11),
            (4, 10),

            # Middle Lines
            (6, 12),
            (2, 9)
        ]

    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), -1)
        out = frame.copy()
        alpha = 0.4
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out
    
    def draw_court(self, frame):
        # Keypoints
        for i in range(0, len(self.drawing_kp), 2):
            x = int(self.drawing_kp[i])
            y = int(self.drawing_kp[i+1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Lines
        for line in self.lines:
            pt1 = (int(self.drawing_kp[(line[0]-1)*2]), int(self.drawing_kp[(line[0]-1)*2 + 1]))
            pt2 = (int(self.drawing_kp[(line[1]-1)*2]), int(self.drawing_kp[(line[1]-1)*2 + 1]))
            cv2.line(frame, pt1, pt2, (0, 0, 0), 2)
        
        # Net
        net_start = (int(self.court_start_x), int((self.court_start_y + self.court_end_y) / 2))
        net_end = (int(self.court_end_x), int((self.court_start_y + self.court_end_y) / 2))
        cv2.line(frame, net_start, net_end, (0, 0, 0), 4)

        return frame


    def draw_all(self, frames):
        output_frames = []

        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)

            output_frames.append(frame)
        
        return output_frames

    def get_diagram_start_point(self):
        return (self.start_x, self.start_y)
    
    def get_diagram_end_point(self):
        return (self.end_x, self.end_y)
    
    def get_diagram_width(self):
        return self.drawing_rectangle_width
    
    def get_diagram_keypoints(self):
        return self.drawing_kp
    
    def map_real_to_diagram(self, player_boxes, ball_boxes, court_keypoints):
        mapped_player_boxes = []
        mapped_ball_boxes = []

        if court_keypoints is None:
            return player_boxes, ball_boxes
        
        # Clean up court keypoints
        temp = []*24
        for i in range(0, 36, 3):
            temp.append(court_keypoints[i])
            temp.append(court_keypoints[i+1])
        
        court_keypoints = temp

        # Get scale and offset using homography
        src_pts = np.array(court_keypoints, dtype=np.float32).reshape(12, 2)
        dst_pts = np.array(self.get_diagram_keypoints(), dtype=np.float32).reshape(12, 2)

        # Compute homography (src = real court, dst = diagram)
        matrix, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC)

        # Map player boxes
        for frame in player_boxes:
            mapped_frame = {}
            for track_id, bbox in frame.items():
                x1, y1, x2, y2 = bbox
                points = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
                transformed_points = cv2.perspectiveTransform(points, matrix)[0]
                x1_new, y1_new = transformed_points[0]
                x2_new, y2_new = transformed_points[1]
                mapped_frame[track_id] = [x1_new, y1_new, x2_new, y2_new]
            mapped_player_boxes.append(mapped_frame)

        # Map ball boxes
        for frame in ball_boxes:
            mapped_frame = {}
            for track_id, bbox in frame.items():
                x1, y1, x2, y2 = bbox
                points = np.array([[[x1, y1], [x2, y2]]], dtype=np.float32)
                transformed_points = cv2.perspectiveTransform(points, matrix)[0]
                x1_new, y1_new = transformed_points[0]
                x2_new, y2_new = transformed_points[1]
                mapped_frame[track_id] = [x1_new, y1_new, x2_new, y2_new]
            mapped_ball_boxes.append(mapped_frame)


        diagram_net_y = (self.court_start_y + self.court_end_y) / 2
    
        # Create points along the net line in the diagram
        net_points_diagram = np.array([[[self.court_start_x, diagram_net_y], 
                                    [self.court_end_x, diagram_net_y]]], dtype=np.float32)
        
        # Use inverse homography to map back to real court
        inverse_matrix = np.linalg.inv(matrix)
        real_net_points = cv2.perspectiveTransform(net_points_diagram, inverse_matrix)[0]
        
        # Get the average y-coordinate of the net in real court
        real_net_y = (real_net_points[0][1] + real_net_points[1][1]) / 2

        return mapped_player_boxes, mapped_ball_boxes, real_net_y
    
    def draw_players_on_diagram(self, frames, diagram_detections, color=(0, 255, 0)):
        output_frames = []

        for frame, detection in zip(frames, diagram_detections):
            for track_id, bbox in detection.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, str(track_id), (int((x1+x2)/2), int(y2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
                # Cricle
                cv2.circle(frame, (int((x1+x2)/2), int(y2)), 5, color, -1)
                # Border
                cv2.circle(frame, (int((x1+x2)/2), int(y2)), 6, (0, 0, 0), 2)

            output_frames.append(frame)
        
        return output_frames
    
    def draw_ball_on_diagram(self, frames, diagram_detections, color=(255, 0, 0)):
        output_frames = []

        for frame, detection in zip(frames, diagram_detections):
            for track_id, bbox in detection.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, str(track_id), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.circle(frame, (int((x1+x2)/2), int((y1+y2)/2)), 5, color, -1)
            output_frames.append(frame)
        
        return output_frames
