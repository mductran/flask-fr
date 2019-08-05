import os
import cv2

from utils.camera import VideoCamera, VideoClassifier
from backend.mtcnn.load_mtcnn import Detection

video_camera = None
global_frame = None

clf = None
le = None


def video_stream():
    global video_camera
    global global_frame

    if not video_camera:
        video_camera = VideoCamera()

    while True:
        frame = video_camera.get_frame()

        if frame:
            global_frame = frame
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # video_camera.capture()
        else:
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
            # video_camera.capture()


def classifier_stream():
    global video_camera
    global global_frame

    if not video_camera:
        video_camera = VideoClassifier()

    while True:
        frame = video_camera.get_frame()

        if frame:
            global_frame = frame
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # video_camera.capture()
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + global_frame + b'\r\n\r\n')
            # video_camera.capture()


def extract_frame(video_path, skip=3):
    video = cv2.VideoCapture(video_path)
    name = video_path.split('.')[0].split('/')[-1]
    stack = []
    count = 0

    find_faces = Detection().find_faces

    while True:
        _, frame = video.read()
        if _:
            count += 1
            if count % skip == 0:
                stack.append(frame)
                if os.path.isdir('tmp/'+name):
                    faces = find_faces(frame)
                    for i in range(len(faces)):
                        cv2.imwrite('tmp/' + name + "/{}_{}.jpg".format(count//skip, i), faces[i].image)
                else:
                    os.mkdir('tmp/'+name)
                    faces = find_faces(frame)
                    for i in range(len(faces)):
                        cv2.imwrite('tmp/' + name + "/{}_{}.jpg".format(count//skip, i), faces[i].image)
        else:
            break
    return stack
