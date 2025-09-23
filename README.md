# Pickle CV

A computer vision project to analyze pickleball games. This project uses object detection and tracking to identify players and the ball, track their movements, and calculate game statistics.

## Key Features
- **Player and Ball Detection:** Utilizes a YOLOv11 model to detect players and the ball in video footage.
- **Court Line Detection:** Identifies the court lines to establish a frame of reference.
- **Movement Tracking:** Tracks the detected objects across frames to analyze movement and game events.
- **Data Output:** Generates an annotated video highlighting detections and can save tracking data.

## Technology Used
- **Python**
- **OpenCV:** For video processing and core computer vision algorithms.
- **PyTorch:** For loading and running the deep learning models.
- **YOLOv11:** For real-time object detection.

## How to Run
1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Place an input video** in the `sample_inputs` directory.
4.  **Run the main analysis script:**
    ```bash
    python main.py
    ```
5.  The processed video will be saved in the `output_videos` directory.
