import os
from ultralytics import YOLO
import cv2

video = cv2.VideoCapture("can_videos\\test_video_5.mp4") # Read the video file
ret, frame = video.read()

H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get the height and width of the video
W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

threshold = 0.8 # Set the threshold for the model

writer = cv2.VideoWriter(f"predictions\\prediction_5_{threshold}.mp4", cv2.VideoWriter_fourcc(*'MP4V'), int(video.get(cv2.CAP_PROP_FPS)), (W, H)) # Create a video writer object

model = YOLO("runs\\detect\\train13\\weights\\best.pt") # Load the model

while ret:
    results = model(frame)[0] # Get the results from the model

    for result in results.boxes.data.tolist(): # Draw the bounding boxes on the frame
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper() + str(score), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3, cv2.LINE_AA)

    writer.write(frame) # Write the frame to the video file
    ret, frame = video.read() # Read the next frame

video.release()
writer.release()
cv2.destroyAllWindows()