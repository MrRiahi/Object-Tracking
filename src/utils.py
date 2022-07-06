import cv2


def save_video(frames, video_name):
    """
    This function save frames as a video
    :param frames:
    :param video_name:
    :return:
    """

    # Initialize the output video file
    file_path = 'output_data/' + video_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(file_path, fourcc, 25, (320, 240))

    # Write video
    for frame in frames:
        video_writer.write(frame)
    video_writer.release()
