from flask import Flask, render_template, Response, jsonify, request
from utils.utils import *
from utils.s3_utils import push

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('collect.html')


@app.route('/record_status', methods=['POST'])
def record_status():
    global video_camera
    if not video_camera:
        video_camera = VideoCamera()

    json = request.get_json()

    status = json['status']

    if status == "true":
        video_camera.start_record()
        return jsonify(result="started")
    else:
        video_camera.stop_record()
        return jsonify(result="stopped")


@app.route('/video_viewer')
def video_viewer():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/switch_page', methods=['POST'])
def switch():
    global video_camera
    video_camera = None
    return render_template('classify.html')


@app.route('/video_classifier')
def video_classifier():
    return Response(classifier_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=['POST'])
def get_eid():
    eid = request.form['EID']
    nname = 'static/{}.mp4'.format(eid)
    to_push = 'tmp/{}'.format(eid)
    if os.path.exists('static/video.avi'):
        try:
            os.rename('static/video.avi', nname)
            extract_frame(nname)
            push(to_push)
            return render_template('collect.html')
        except FileExistsError:
            return render_template('collect.html')
