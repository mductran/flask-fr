import os
import cv2
import numpy as np
from keras.models import load_model
import keras.backend as backend
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import logging


def prewhiten(x):
    if x.ndim == 4:
        axis = (1, 2, 3)
        size = x[0].size
    elif x.ndim == 3:
        axis = (0, 1, 2)
        size = x.size
    else:
        raise ValueError('Dimension should be 3 or 4')

    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0 / np.sqrt(size))
    y = (x - mean) / std_adj
    return y


def stack_images(path):
    stack = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.split('.')[-1] == 'jpg' or file.split('.')[-1] == 'png':
                img = cv2.imread(os.path.join(root, file))
                img = cv2.resize(img, (160, 160))
                img = prewhiten(img)
                stack.append(img)
    print('Stacked {} images'.format(len(stack)))
    return np.stack(stack)


def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output


def read_classes(aligned_face_path):
    ignore = ['.git', '.idea', '__pycache__']
    class_list = []
    for name in os.listdir(aligned_face_path):
        if os.path.isdir(os.path.join(aligned_face_path, name)):
            if name not in ignore:
                class_list.append(name)
    return class_list


def read_names(path):
    names = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            names.append(name)
    return names


def draw(img, mode='mtcnn', detect_fn=None, model=None):
    modes = ['haar', 'mtcnn']
    if mode == modes[0]:
        box = []
        haar = cv2.CascadeClassifier("model_data/haarcascade_frontalface_default.xml")
        faces = haar.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            box.append((x, y, w, h))
        return box
    elif mode == modes[1]:
        box = []
        faces = detect_fn(img)
        for face in faces:
            box.append(face.bounding_box)
            x, y, w, h = face.bounding_box
            cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        return box
    else:
        raise ValueError("Invalid mode. Expected one of: %s" % modes)


def calculate_embeddings(model, file, mode='single', batch_size=32):
    modes = ['single', 'folder', 'batch']
    if mode == modes[0]:
        if type(file) is str:
            img = cv2.imread(file)
        else:
            img = file
        if not img.shape:
            raise ValueError("Image path '{}' does not exist".format(file))
        img = prewhiten(img)
        if len(img.shape) == 3:
            img = img[np.newaxis]
        return model.predict(img)
    elif mode == modes[1]:
        img = stack_images(file)
        pred = []
        for start in tqdm(range(0, len(img), batch_size)):
            pred.append(model.predict_on_batch(img[start:start+batch_size]))
        embs = l2_normalize(np.concatenate(pred))
        return embs
    elif mode == modes[2]:
        aligned = []
        pd = []
        for path in file:
            img = cv2.imread(path)
            aligned.append(img)
        aligned = np.array(aligned)
        for start in range(0, len(aligned), batch_size):
            pd.append(model.predict_on_batch(aligned[start:start + batch_size]))
        embs = l2_normalize(np.concatenate(pd))
        return embs
    else:
        raise ValueError("Invalid mode. Expected one of: %s" % modes)


def train_classifier(model, basepath, max_img=20):
    labels = []
    embs = calculate_embeddings(model, basepath, mode='folder')
    names = read_names(basepath)
    for name in names:
        labels.extend([name]*len(os.listdir(basepath+'/'+name)))
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    clf = SVC(kernel='linear', probability=True).fit(embs, y)

    return le, clf


def visualize_embeddings(model, filepath):
    names = read_names(filepath)
    data = {}
    d = {}
    for name in names:
        d[name] = []
        image_dirpath = filepath + '/' + name
        image_filepaths = [image_dirpath+'/'+f for f in os.listdir(image_dirpath)]
        for i in range(len(image_filepaths)):
            img = cv2.imread(image_filepaths[i])
            if img.shape != (160, 160, 3):
                img = cv2.resize(img, (160, 160))
            emb = calculate_embeddings(model, img, 'single')
            data['{}{}'.format(name, i)] = {'image_filepath': image_filepaths[i],
                                            'emb': emb}
            d[name].append(emb.flatten())

    x = []
    for v in data.values():
        x.append(v['emb'].flatten())
    # keep 3 components for 3d plot
    pca = PCA(n_components=3).fit(x)

    for key in d:
        d[key] = pca.transform(d[key])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10

    for key in d:
        ax.plot(d[key][:, 0], d[key][:, 1], d[key][:, 2], 'o',
                markersize=8, color=np.random.rand(3,),
                alpha=0.5, label=key)

    plt.title("Embedding Vector")
    ax.legend(loc='upper right')
    plt.show()


def classify(model, classifier, label_encoder, image):
    if type(image) is str:
        image = cv2.imread(image)
    image = cv2.resize(image, (160, 160))
    emb = calculate_embeddings(model, image[np.newaxis], mode='single')
    try:
        name = label_encoder.inverse_transform(classifier.predict(emb))
    except:
        name = None
    return name


if __name__ == '__main__':
    model_ = load_model('model_data/model.h5')
    le_, clf_ = train_classifier(model_, '../../tmp')
    import time
    start = time.time()
    print(classify(model_, clf_, le_, 'd:/datasets/face/6/ductm22/VID_20190701_130354_125016_0.png'))
    print(time.time() - start)
