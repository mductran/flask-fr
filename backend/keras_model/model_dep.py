import os
import cv2
import numpy as np
from keras.models import load_model
import keras.backend as K
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d


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


def embed_path(model, image_stack, batch_size=32):
    pd = []
    for start in range(0, len(image_stack), batch_size):
        pd.append(model.predict_on_batch(image_stack[start:start + batch_size]))
    embs = K.l2_normalize(np.concatenate(pd))

    return embs


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


def calc_embs(model, filepaths, batch_size=32):
    aligned_images = stack_images(filepaths)
    pd = []
    for start in tqdm(range(0, len(aligned_images), batch_size)):
        pd.append(model.predict_on_batch(aligned_images[start:start + batch_size]))
    embs = l2_normalize(np.concatenate(pd))
    return embs


def calc_embs_vis(model, img):
    img = prewhiten(img)
    if len(img.shape) == 3:
        img = img[np.newaxis]
    return model.predict(img)


def calc_embs_one(model, filepath, batch_size=32):
    aligned_images = stack_images(filepath)
    pd = []
    for start in range(0, len(aligned_images), batch_size):
        pd.append(model.predict_on_batch(aligned_images[start:start + batch_size]))
    embs = l2_normalize(np.concatenate(pd))

    return embs


def train_classifier(model, dir_basepath):
    labels = []
    data = {}

    # for name in names:
    #     dirpath = os.path.abspath(dir_basepath + name)
    #     print(dirpath)
    #     filepaths = [os.path.join(dirpath, f) for f in os.listdir(dirpath)][:max_num_img]
    #     print("filepaths: ", filepaths)
    #     embs_ = calc_embs(model, filepaths)
    #     labels.extend([name] * len(embs_))
    #     embs.append(embs_)

    print('calculating embeddings\n')

    embs = calc_embs(model, dir_basepath)
    names = get_names(dir_basepath)
    for name in names:
        labels.extend([name] * len(os.listdir(dir_basepath + '/' + name)))
    with open('labels.txt', 'w') as f:
        for label in labels:
            f.write(label+'\n')
    print('===\ndata preparation done\n===\n')

    # embs = np.concatenate(embs)
    le = LabelEncoder().fit(labels)
    y = le.transform(labels)
    print(type(y))
    clf = SVC(kernel='linear', probability=True).fit(embs, y)

    return le, clf


def get_names(path):
    names = []
    for root, dirs, files in os.walk(path):
        for name in dirs:
            names.append(name)
    return names


def visualize_embeddings(model, filepath):
    names = get_names(filepath)
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
            emb = calc_embs_vis(model, img)
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


def visualize_embeddings_(model, filepath):
    names = get_names(filepath)
    data = {}

    for name in names:
        image_dirpath = filepath + '/' + name
        image_filepaths = [image_dirpath+'/'+f for f in os.listdir(image_dirpath)]
        embs = calc_embs_one(model, image_dirpath)
        print(embs.shape)
        for i in range(len(image_filepaths)):
            data['{}{}'.format(name, i)] = {'image_filepath': image_filepaths[i],
                                            'emb': embs}

    x = []
    for v in data.values():
        for _ in v['emb']:
            x.append(_)
    # keep 3 components for 3d plot
    pca = PCA(n_components=3).fit(x)

    d = {}
    for k, v in data.items():
        d[k] = pca.transform(v['emb'])

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
    image = cv2.resize(image, (160, 160))
    emb = calc_embs_vis(model, image[np.newaxis])
    try:
        name = label_encoder.inverse_transform(classifier.predict(emb))
    except:
        name = None
    return name


def draw(img, detect_fn=None, model=None):
    haar = cv2.CascadeClassifier("model_data/haarcascade_frontalface_default.xml")
    faces = haar.detectMultiScale(img, 1.3, 5)
    box = []
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        box.append((x, y, w, h))
    # box = []
    # faces = detect_fn(img)
    # for face in faces:
    #     box.append(face.bounding_box)
    #     x, y, w, h = face.bounding_box
    #     cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
    return box


def show(data_dir):
    model = load_model('model_data/model.h5')
    label_encoder, classifier = train_classifier(model, data_dir)

    cap = cv2.VideoCapture(0)
    while True:
        _, frame = cap.read()
        boxes = draw(frame)
        for box in boxes:
            canvas = frame[box[0]:box[0]+box[2], box[1]:box[1]+box[3]]
            name = classify(model, classifier, label_encoder, canvas)
            cv2.putText(canvas, name, (box[0], box[2]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('res', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show('../../tmp')
