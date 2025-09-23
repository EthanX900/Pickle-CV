from ultralytics import YOLO
import cv2
import pickle
from utils.bbox_utils import get_center_bbox, measure_distance

class PlayerTracker:
    def __init__(self, model_path="models/keypoints_model.pt"):
        self.model = YOLO(model_path)

    def detect_frame(self, frame):
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results[0].names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = int(box.cls.tolist()[0])
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        
        return player_dict
    
    def detect_frames(self, frames, read_from_save = False, save_path = None):
        all_player_dicts = []

        if read_from_save and save_path is not None:
            with open(save_path, 'rb') as f:
                all_player_dicts = pickle.load(f)
            return all_player_dicts

        for frame in frames:
            player_dict = self.detect_frame(frame)
            all_player_dicts.append(player_dict)

        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(all_player_dicts, f)

        return all_player_dicts
    
    def draw_boxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2, = bbox
                cv2.putText(frame, str(track_id), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            output_video_frames.append(frame)
        
        return output_video_frames
    
    def choose_players(self, court_keypoints, player_dict, num_players=4):
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 3):
                court_point = (court_keypoints[i], court_keypoints[i+1])
                distance = measure_distance(player_center, court_point)
                if distance < min_distance:
                    min_distance = distance
        
            distances.append((track_id, min_distance))
        
        distances.sort(key=lambda x: x[1])

        chosen_players = [track_id for track_id, _ in distances[:num_players]]
        return chosen_players



    def filter_players(self, court_keypoints, player_detections, num_players=4):
        player_detection = player_detections[0]
        chosen_players = self.choose_players(court_keypoints, player_detection, num_players=4)

        filtered_detections = []
        for player_dict in player_detections:
            filtered_dict = {track_id: bbox for track_id, bbox in player_dict.items() if track_id in chosen_players}
            filtered_detections.append(filtered_dict)

        return filtered_detections