# program to read amplified frames,
# and save them as amplified video file

import argparse
import cv2
import os
import sys

sys.path.append('C:/Users/282780/deep_motion_mag')

parser = argparse.ArgumentParser(description='')
parser.add_argument('--origVid_path', default='data/input/baby/baby.mp4',
                    help='path to original video file')
parser.add_argument('--framesDir_path', default='data/output/baby_frames_amp10_static',
                    help='path to amplified frames')
parser.add_argument('--vidDir_path', default='data/output/baby_video_amp10_static',
                    help='path to amplified video')
arguments = parser.parse_args()

# Function to extract frames
def VideoCreature(args):
    # create VideoCapture object from video file
    orig_vid = cv2.VideoCapture(args.origVid_path)

    if not orig_vid.isOpened():
        print("Error reading video file")

    # extract original video properties
    frame_width = int(orig_vid.get(3))
    frame_height = int(orig_vid.get(4))
    size = (frame_width, frame_height)
    FPS = orig_vid.get(5)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec code (i.e., h264/h265/MJPG code)

    if not os.path.exists(args.vidDir_path):
        os.makedirs(args.vidDir_path)
    vid_name = 'baby_' + args.vidDir_path.split("video_")[-1] + '.mp4'

    # create VideoWriter object to save video file
    amp_vid = cv2.VideoWriter(args.vidDir_path + "/" + vid_name, fourcc, FPS, size)
    print(args.vidDir_path + "/" + vid_name)

    # Path to video file
    frames_dir = os.listdir(args.framesDir_path)  # list of files (.png) names
    frames_dir.sort(key=len)  # sort by "0, 1, 2, 3, ..." and not by "0, 1, 10, 100, ..."

    for frame_name in frames_dir:
        # read image
        frame = cv2.imread(args.framesDir_path + '/' + frame_name)

        # Write the frame into the .mp4 file
        amp_vid.write(frame)

        # Display the frame saved in the file
        cv2.imshow('baby_amplified_video', frame)

        # play video at ~30FPS, press S on keyboard to stop the process
        if cv2.waitKey(33) & 0xFF == ord('s'):
            break

    # When everything done, release the video capture and video write objects
    amp_vid.release()
    orig_vid.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    print("The amplified video was successfully saved")


# Driver Code
if __name__ == '__main__':
    # Calling the function
    VideoCreature(arguments)
