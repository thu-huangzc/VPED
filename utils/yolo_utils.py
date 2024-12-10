import cv2

def save_yolo_tracking_video(results, input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # 获取视频的帧率和分辨率
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    # 定义输出视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for result in results:
        frame = result.orig_img
        frame_with_bbox = result.plot(img=frame)
        out.write(frame_with_bbox)

    # 释放资源
    out.release()
    cv2.destroyAllWindows()
