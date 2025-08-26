import cv2
import os

def generate_frames(video_path, output_folder, frame_rate=1):
    """
    Extract frames from a video file and save them to a specified folder.

    :param video_path: Path to the input video file.
    :param output_folder: Folder where extracted frames will be saved.
    :param frame_rate: Number of frames to extract per second.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        if frame_count  % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:06d}.PNG")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        
        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} to {output_folder}.")

if __name__ == "__main__":
    #extract frames for all video in a folder
    folder_path = "./football_train"
    for sub_folder in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder)
        if os.path.isdir(sub_folder_path):
            for video_file in os.listdir(sub_folder_path):
                if video_file.endswith(".mp4"):
                    video_path = os.path.join(sub_folder_path, video_file)
                    output_folder = os.path.join("extracted_frames", sub_folder)
                    generate_frames(video_path, output_folder, frame_rate=1)
    # video_path = "./football_train/Match_1824_1_0_subclip_3/Match_1824_1_0_subclip_3.mp4"  # Replace with your video file path
    # output_folder = "extracted_frames"  # Replace with your desired output folder
    # generate_frames(video_path, output_folder, frame_rate=1)