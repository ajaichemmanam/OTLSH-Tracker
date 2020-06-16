# Main Tracking
import cv2

from tracker import Tracker
# from tfgraphdetector import ObjectDetector
from objectDetector.centerNet.centernetDetector import ObjectDetector

from visualization import plot_detections, plot_tracks, plot_trajectory
import imutils.video

fps_imutils = imutils.video.FPS().start()


# Initialise Object Detector

PATH_TO_FROZEN_GRAPH = ''
PATH_TO_LABELS = ''
PATH_TO_ENCODER_MODEL = ''
detector = ObjectDetector(model_path=PATH_TO_FROZEN_GRAPH,
                          label_path=PATH_TO_LABELS,
                          encoder_path=PATH_TO_ENCODER_MODEL,
                          min_conf=0.7)


# Initialise Tracker
tracker = Tracker(frame_rate=30, lsh_mode=0.5)
frame_id = 0

# Debug
writeVideo_flag = True

# Read Video
PATH_TO_VIDEO = ''
cap = cv2.VideoCapture(PATH_TO_VIDEO)

# Save Video
if writeVideo_flag:
    w = int(cap.get(3))
    h = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30, (w, h))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Object Detection
    detections = detector.inferImage(frame)

    # Tracking
    active_Tracks = tracker.updateTracks(detections)

    fps_imutils.update()  # FPS Update

    # Visualise

    detectionBoxes = [det.tlbr for det in detections]
    detectionScores = [det.score for det in detections]
    frame = plot_detections(frame, detectionBoxes,
                            detectionScores)     # View Detections

    active_boxes = [t.tlwh for t in active_Tracks]
    active_ids = [t.track_id for t in active_Tracks]
    frame = plot_tracks(frame, active_boxes, active_ids,
                        frame_id=frame_id)     # View Tracks

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if writeVideo_flag:  # Save Video Flag
        out.write(frame)  # save a frame

    frame_id += 1

# End
detector.close()
fps_imutils.stop()
print('FPS: {}'.format(fps_imutils.fps()))
cap.release()
cv2.destroyAllWindows()
