import numpy as np
import cv2

def draw_team_stats(output_video_frames,team_stats):

    for index, row in team_stats.iterrows():
        team_1_shot_speed = row['team_1_last_shot_speed_mph']
        team_2_shot_speed = row['team_2_last_shot_speed_mph']

        avg_team_1_shot_speed = row['team_1_average_shot_speed_mph']
        avg_team_2_shot_speed = row['team_2_average_shot_speed_mph']


        frame = output_video_frames[index]
        shapes = np.zeros_like(frame, np.uint8)

        width=350
        height=230

        start_x = frame.shape[1]-400
        start_y = frame.shape[0]-500
        end_x = start_x+width
        end_y = start_y+height

        overlay = frame.copy()
        cv2.rectangle(overlay, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
        alpha = 0.5 
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        output_video_frames[index] = frame

        text = "     Team 1     Team 2"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+80, start_y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        text = "Shot Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{team_1_shot_speed:.1f} mph    {team_2_shot_speed:.1f} mph"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        text = "avg. S. Speed"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+10, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        text = f"{avg_team_1_shot_speed:.1f} mph    {avg_team_2_shot_speed:.1f} mph"
        output_video_frames[index] = cv2.putText(output_video_frames[index], text, (start_x+130, start_y+160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return output_video_frames