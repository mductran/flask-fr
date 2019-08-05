import threading
import keras.backend as backend
from backend.keras_model.model_dep import *
from backend.mtcnn.load_mtcnn import Detection


class RecordingThread(threading.Thread):
    def __init__(self, name, camera):
        threading.Thread.__init__(self)
        self.name = name
        self.isRunning = True

        self.cap = camera
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('./static/video.avi', fourcc, 20.0, (640, 480))

    def run(self):
        while self.isRunning:
            ret, frame = self.cap.read()
            if ret:
                self.out.write(frame)

        self.out.release()

    def stop(self):
        self.isRunning = False

    def __del__(self):
        self.out.release()


class VideoCamera(object):
    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture(0)

        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None

        backend.clear_session()
        self.detect_fn = Detection().find_faces
        # self.cascade = cv2.CascadeClassifier('keras_model/model_data/haarcascade_frontalface_default.xml')

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        faces = self.detect_fn(frame)
        for face in faces:
            x, y, w, h = face.bounding_box
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 0), 2)

        # faces = self.cascade.detectMultiScale(frame, 1.3, 5)
        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)

            # Record video
            # if self.is_record:
            #     if self.out == None:
            #         fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            #         self.out = cv2.VideoWriter('./static/video.avi',fourcc, 20.0, (640,480))

            #     ret, frame = self.cap.read()
            #     if ret:
            #         self.out.write(frame)
            # else:
            #     if self.out != None:
            #         self.out.release()
            #         self.out = None  

            return jpeg.tobytes()

        else:
            return None

    def start_record(self):
        self.is_record = True
        self.recordingThread = RecordingThread("Video Recording Thread", self.cap)
        self.recordingThread.start()

    def stop_record(self):
        self.is_record = False

        if self.recordingThread:
            self.recordingThread.stop()


class VideoClassifier(object):
    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture(0)

        # Initialize video recording environment
        self.is_record = False
        self.out = None

        # Thread for recording
        self.recordingThread = None

        backend.clear_session()
        self.detect_fn = Detection().find_faces
        # self.cascade = cv2.CascadeClassifier('keras_model/model_data/haarcascade_frontalface_default.xml')

        self.model = load_model('keras_model/model_data/model.h5')
        self.le, self.clf = train_classifier(self.model, 'tmp')

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        faces = self.detect_fn(frame)
        for face in faces:
            x, y, w, h = face.bounding_box
            cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
            img = frame[y:y+h, x:x+w]
            name = classify(self.model, self.clf, self.le, img)[0]
            cv2.putText(frame, name, (x, h), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1, (0, 255, 0), 2, 2)

        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()

        else:
            return None
