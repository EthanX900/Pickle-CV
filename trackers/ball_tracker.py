from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path="models/keypoints_model.pt"):
        self.model = YOLO(model_path)

    def detect_frame(self, frame, previous_detections=None):
        results = self.model.predict(frame)[0]
        ball_dict = {}
        
        # If there are detections
        if len(results.boxes) > 0:
            result = results.boxes.xyxy.tolist()[0]
            ball_dict[1] = result
            
            # If we have previous detections, validate the current detection
            if previous_detections:
                # Check if this is a valid detection based on previous positions
                is_valid = self._validate_detection(result, previous_detections)
                if not is_valid:
                    # Return empty dictionary if detection is not valid
                    return {}
        
        return ball_dict
    
    def _validate_detection(self, current_detection, previous_detections):

        if not previous_detections:
            return True
            
        # Get the most recent valid detection
        most_recent = None
        frames_between = 1
        
        # Find the most recent non-empty detection
        for i, detection in enumerate(previous_detections):
            if detection:  # If detection is not empty
                most_recent = list(detection.values())[0]  # Get the bbox values
                frames_between = i + 1
                break
                
        if most_recent is None:
            return True
            
        # Calculate center points
        current_center_x = (current_detection[0] + current_detection[2]) / 2
        current_center_y = (current_detection[1] + current_detection[3]) / 2
        recent_center_x = (most_recent[0] + most_recent[2]) / 2
        recent_center_y = (most_recent[1] + most_recent[3]) / 2
        
        # Calculate distance between centers
        distance = ((current_center_x - recent_center_x) ** 2 + 
                   (current_center_y - recent_center_y) ** 2) ** 0.5
        
        # pixel threshold
        x_pixels = 5
        # Check if distance is within threshold adjusted by frames between
        return distance <= (x_pixels * frames_between)
    
    def detect_frames(self, frames, read_from_save = False, save_path = None):
        ball_dicts = []

        if read_from_save and save_path is not None:
            with open(save_path, 'rb') as f:
                ball_dicts = pickle.load(f)
            return ball_dicts

        for i, frame in enumerate(frames):
            # Get previous 5 detections (or fewer if not available)
            previous_detections = ball_dicts[max(0, i-5):i][::-1]  # Most recent first
            ball_dict = self.detect_frame(frame, previous_detections)
            ball_dicts.append(ball_dict)

        if save_path is not None:
            with open(save_path, 'wb') as f:
                pickle.dump(ball_dicts, f)

        return ball_dicts
    
    def interpolate_ball_detections(self, ball_positions):
        ball_positions = [x.get(1, []) for x in ball_positions]
        df_ball = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball = df_ball.interpolate()
        df_ball = df_ball.bfill()

        ball_positions = [{1:x} for x in df_ball.to_numpy().tolist()]

        return ball_positions
    
    def get_hit_frames(self, ball_positions, net_y):
        
        ball_positions = [x.get(1,[]) for x in ball_positions]
        # convert the list into pandas dataframe
        df_ball = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball['ball_hit'] = 0

        df_ball['mid_y'] = (df_ball['y1'] + df_ball['y2'])/2
        df_ball['mid_y_rolling_mean'] = df_ball['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball['delta_y'] = df_ball['mid_y_rolling_mean'].diff()
        
        # Add net position labeling
        df_ball['position_relative_to_net'] = df_ball['mid_y'].apply(
            lambda y: 'above' if y < net_y else 'below'
        )
        
        # Track net crossings
        df_ball['crossed_net'] = False
        for i in range(1, len(df_ball)):
            if df_ball['position_relative_to_net'].iloc[i-1] != df_ball['position_relative_to_net'].iloc[i]:
                df_ball.loc[i, 'crossed_net'] = True

        minimum_change_frames_for_hit = 10
        for i in range(1,len(df_ball)- int(minimum_change_frames_for_hit*1.2) ):
            negative_position_change = df_ball['delta_y'].iloc[i] >0 and df_ball['delta_y'].iloc[i+1] <0
            positive_position_change = df_ball['delta_y'].iloc[i] <0 and df_ball['delta_y'].iloc[i+1] >0

            if negative_position_change or positive_position_change:
                change_count = 0 
                for change_frame in range(i+1, i+int(minimum_change_frames_for_hit*1.2)+1):
                    negative_position_change_following_frame = df_ball['delta_y'].iloc[i] >0 and df_ball['delta_y'].iloc[change_frame] <0
                    positive_position_change_following_frame = df_ball['delta_y'].iloc[i] <0 and df_ball['delta_y'].iloc[change_frame] >0

                    if negative_position_change and negative_position_change_following_frame:
                        change_count+=1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count+=1
            
                if change_count>minimum_change_frames_for_hit-1:
                    # Get current ball position relative to net
                    current_side = df_ball['position_relative_to_net'].iloc[i]
                    
                    # Find previous valid hits
                    previous_hits = df_ball[df_ball['ball_hit'] == 1]
                    
                    if len(previous_hits) == 0:
                        # First hit - no validation needed
                        df_ball['ball_hit'].iloc[i] = 1
                    else:
                        # Get the side of the most recent previous hit
                        last_hit_index = previous_hits.index[-1]
                        last_hit_side = df_ball['position_relative_to_net'].iloc[last_hit_index]
                        
                        # Only count as hit if on opposite side from previous hit
                        if current_side != last_hit_side:
                            df_ball['ball_hit'].iloc[i] = 1

        frame_nums_with_ball_hits = df_ball[df_ball['ball_hit']==1].index.tolist()

        print("Ball hit frames:", frame_nums_with_ball_hits)

        return frame_nums_with_ball_hits
            
    
    def draw_boxes(self, video_frames, ball_detections):
        output_video_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2, = bbox
                cv2.putText(frame, str(track_id), (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                
            output_video_frames.append(frame)
        
        return output_video_frames
        