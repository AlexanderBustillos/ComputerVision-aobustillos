import cv2

import numpy as np


def process_videos():
    cap_base = cv2.VideoCapture("CodeVideo.mov")
    cap_overlay = cv2.VideoCapture("VideoOverlay.mov")

    fps_base = cap_base.get(cv2.CAP_PROP_FPS)
    fps_overlay = cap_overlay.get(cv2.CAP_PROP_FPS)
    fps = min(fps_base, fps_overlay)
    width = int(cap_base.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_base.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output_video.mp4", fourcc, fps, (width, height))
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    frame_count = 0
    last_valid_H = None
    markers_missing_frames = 0
    max_missing_frames = 100

    while True:
        ret1, frame1 = cap_base.read()
        ret2, frame2 = cap_overlay.read()

        if not ret1:
            break

        if not ret2:
            cap_overlay.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret2, frame2 = cap_overlay.read()
            if not ret2:
                break

        frame_count += 1

        corners, ids, _ = detector.detectMarkers(frame1)

        markers_detected = False
        if ids is not None:
            ids = ids.flatten()
            required_markers = [0, 1, 2, 3]
            if all(m in ids for m in required_markers):
                markers_detected = True
                markers_missing_frames = 0

                marker_pos = {}
                for corner, marker_id in zip(corners, ids):
                    if marker_id in required_markers:
                        marker_pos[marker_id] = corner.reshape(4, 2)

                h, w = frame2.shape[:2]
                src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)

                dst = np.array([
                    marker_pos[0][0],
                    marker_pos[1][1],
                    marker_pos[2][2],
                    marker_pos[3][3]
                ], np.float32)

                H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

                if H is not None:
                    last_valid_H = H

        if not markers_detected:
            markers_missing_frames += 1
            if markers_missing_frames > max_missing_frames:
                last_valid_H = None

        if last_valid_H is not None:
            h, w = frame2.shape[:2]
            warped = cv2.warpPerspective(frame2, last_valid_H, (width, height))

            gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.GaussianBlur(mask, (5, 5), 1)

            mask_float = mask.astype(float) / 255.0
            mask_3ch = cv2.merge([mask_float, mask_float, mask_float])

            frame1 = (frame1 * (1 - mask_3ch) + warped * mask_3ch).astype(np.uint8)

        out.write(frame1)

    cap_base.release()
    cap_overlay.release()
    out.release()
    cv2.destroyAllWindows()


process_videos()