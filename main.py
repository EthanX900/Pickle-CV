from utils import (read_video, 
                   save_video, convert_pixel_distance_to_feet, convert_feet_to_pixel_distance, measure_distance, draw_team_stats)

from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from court_diagram import CourtDiagram
import cv2
import constants
import pandas as pd
from copy import deepcopy

def main():
    # Read in video
    input_video_path = "sample_inputs/clip5.mp4"
    video_frames = read_video(input_video_path)
    
    # Get frame rate
    video = cv2.VideoCapture(input_video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {fps}")
    video.release()

    # Detecting players
    player_tracker = PlayerTracker(model_path = "models/yolo11m.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_save=True, save_path="tracker_saves/player_detections.pkl")

    # Initialize Court
    court_diagram = CourtDiagram(video_frames[0])

    # Detecting balls
    ball_tracker = BallTracker(model_path= "models/ball_tracker_v10.pt")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_save=True, save_path="tracker_saves/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_detections(ball_detections)

    # Detecting court keypoints
    court_line_detector = CourtLineDetector(model_path="models/keypoints_model_v6.pth")
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Draw Diagram
    video_frames = court_diagram.draw_all(video_frames)


    # Filtering players to only show closest 4 to court (Which should be the players)
    player_detections = player_tracker.filter_players(court_keypoints, player_detections, num_players=4)

    # Convert real to diagram coordinates
    player_diagram_detections, ball_diagram_detections, net_y_position = court_diagram.map_real_to_diagram(player_detections, ball_detections, court_keypoints)
    video_frames = court_diagram.draw_players_on_diagram(video_frames, player_diagram_detections)
    video_frames = court_diagram.draw_ball_on_diagram(video_frames, ball_diagram_detections)

    
    # Get ball hit frames
    
    ball_hit_frames = ball_tracker.get_hit_frames(ball_detections, net_y_position)

    team_stats_data = [{
        'frame': 0,

        'team_1_number_of_shots': 0,
        'team_1_total_shot_speed_mph': 0,
        'team_1_average_shot_speed_mph': 0,
        'team_1_last_shot_speed_mph': 0,

        'team_2_number_of_shots': 0,
        'team_2_total_shot_speed_mph': 0,
        'team_2_average_shot_speed_mph': 0,
        'team_2_last_shot_speed_mph': 0
    }]

    for ball_shot_ind in range(len(ball_hit_frames)-1):
        start_frame = ball_hit_frames[ball_shot_ind]
        end_frame = ball_hit_frames[ball_shot_ind+1]
        ball_shot_time_in_seconds = (end_frame-start_frame)/fps

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = measure_distance(ball_diagram_detections[start_frame][1],
                                                           ball_diagram_detections[end_frame][1])
        distance_covered_by_ball_feet = convert_pixel_distance_to_feet( distance_covered_by_ball_pixels,
                                                                        court_diagram.get_diagram_width(),
                                                                           constants.CELL_WIDTH * 2
                                                                           ) 
        
        # Calculate speed
        ball_speed_mph = (distance_covered_by_ball_feet / ball_shot_time_in_seconds) * (3600/5280)

        # Team that hit the ball
        player_positions = player_diagram_detections[start_frame]
        player_shot_ball = min( player_positions.keys(), key=lambda player_id: measure_distance(player_positions[player_id],
                                                                                                 ball_diagram_detections[start_frame][1]))

        if player_shot_ball in [1, 2]:
            player_shot_ball = 1
        else:
            player_shot_ball = 2

        current_team_stats= deepcopy(team_stats_data[-1])
        current_team_stats['frame_num'] = start_frame
        current_team_stats[f'team_{player_shot_ball}_number_of_shots'] += 1
        current_team_stats[f'team_{player_shot_ball}_total_shot_speed_mph'] += ball_speed_mph
        current_team_stats[f'team_{player_shot_ball}_last_shot_speed_mph'] = ball_speed_mph

        team_stats_data.append(current_team_stats)

    team_stats_data_df = pd.DataFrame(team_stats_data)
    frames_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    team_stats_data_df = pd.merge(frames_df, team_stats_data_df, on='frame_num', how='left')
    team_stats_data_df = team_stats_data_df.ffill()

    team_stats_data_df['team_1_average_shot_speed_mph'] = team_stats_data_df['team_1_total_shot_speed_mph']/team_stats_data_df['team_1_number_of_shots']
    team_stats_data_df['team_2_average_shot_speed_mph'] = team_stats_data_df['team_2_total_shot_speed_mph']/team_stats_data_df['team_2_number_of_shots']

    # Draw team stats
    video_frames = draw_team_stats(video_frames, team_stats_data_df)

    # Drawing bounding boxes
    video_frames = player_tracker.draw_boxes(video_frames, player_detections)
    video_frames = ball_tracker.draw_boxes(video_frames, ball_detections)
    video_frames = court_line_detector.draw_keypoints_video(video_frames, court_keypoints)

    # Annotate frame
    for i, frame in enumerate(video_frames):
        cv2.putText(frame, f"Frame: {i}",(10,30),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) 
    

    # Save video
    save_video(video_frames, "output_videos/output_video.mp4", fps=fps)
    
if __name__ == "__main__":
    main()