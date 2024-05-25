import os
import cv2

# This script reads the video files and writes every nth frame to the directory
N = 4
cams = [cv2.VideoCapture("can_videos/can1.mp4"), cv2.VideoCapture("can_videos/can2.mp4"), cv2.VideoCapture("can_videos/can3.mp4")] # Create a list of video capture objects

try: # Create a directory to store the images
    if not os.path.exists("can_images"):
        os.makedirs("can_images")
except OSError:
    print("Error: Cannot create directory")

current_frame = 0
written_frames = 0
for cam in cams:
    while True:
        ret, frame = cam.read() # Read the frame
        if not ret: # If there is no frame left in video, break the loop
            break
        path = "can_images/" + str(written_frames) + "_frame.jpg"
        if current_frame % N == 0: # Write every nth frame to the directory
            cv2.imwrite(path, frame) # Write the frame to the directory
            written_frames += 1
            print(f"{written_frames} written")
        current_frame += 1

cam.release()
cv2.destroyAllWindows()