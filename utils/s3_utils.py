import base64
import boto3
import cv2
import os
from botocore.exceptions import ClientError
import logging


boto3.setup_default_session(region_name='us-west-2')
rekognition = boto3.client('rekognition',
                           aws_access_key_id=base64.b64decode(b'QUtJQVpTSkhVNVBSVVdYM01ZT1I=').decode(),
                           aws_secret_access_key=base64.b64decode(
                               b'dmUvcUNqc1lBRU0yYlF1ekVzOExFeEpuNXBraXZ5QVUxd2dwUTFEQw==').decode())
s3 = boto3.client('s3',
                  aws_access_key_id=base64.b64decode(b'QUtJQVpTSkhVNVBSVVdYM01ZT1I=').decode(),
                  aws_secret_access_key=base64.b64decode(
                      b'dmUvcUNqc1lBRU0yYlF1ekVzOExFeEpuNXBraXZ5QVUxd2dwUTFEQw==').decode()
                  )
bucket = 'flask-tmp'


def read_key(path):
    # TODO: use more secure encoding method
    with open(path) as f:
        e_key = f.readlines()
    access_key = base64.b64decode(e_key[0]).decode()
    secret_key = base64.b64decode(e_key[1]).decode()
    return access_key, secret_key


def extract_frame(video_path, skip=6):
    video = cv2.VideoCapture(video_path)
    name = video_path.split('.')[0].split('/')[-1]
    stack = []
    count = 0
    while True:
        _, frame = video.read()
        if _:
            count += 1
            if count % skip == 0:
                stack.append(frame)
                if os.path.isdir("tmp/" + name):
                    cv2.imwrite("tmp/" + name + "/{}.jpg".format(count//skip), frame)
                else:
                    os.mkdir("tmp/"+name)
                    cv2.imwrite("tmp/" + name + "/{}.jpg".format(count//skip), frame)
        else:
            break
    return stack


def push(folder):
    global s3, bucket
    for root, dirs, files in os.walk(folder):
        dirname = str(root).split('/')[-1]
        for file in files:
            try:
                obj_name = dirname + '/' + file
                print(dirname+'/'+file)
                s3.upload_file(os.path.join(root, file), bucket, obj_name)
            except ClientError as e:
                logging.error(e)


if __name__ == '__main__':
    # push("d:/datasets/face/dcs/images/anhnn5")
    stack = extract_frame('static/ductran.mp4')
