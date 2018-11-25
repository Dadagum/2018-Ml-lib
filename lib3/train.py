from PIL import Image
import numpy as np
import os
from feature import NPDFeature
import pickle
from ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# 原图片路径和处理后保存的图片路径
face_path = "./datasets/original/face"
non_face_path = "./datasets/original/nonface"
grep_face_path = "./datasets/grep/face"
grep_non_face_path = "./datasets/grep/nonface"
# npd特征保存文件位置
# face_npd_file = "D:/testing/python/face.dat"
# non_face_npd_file = "D:/testing/python/non_face.dat"
train_npd_path = "D:/testing/python/train.dat"
valid_npd_path = "D:/testing/python/valid.dat"


def jpg_to_grep(path_from, path_to):
    image_size = (24, 24)
    for f in os.listdir(path_from):
        I = Image.open(path_from + "/" + f)
        I = I.resize(image_size, Image.ANTIALIAS)
        L = I.convert("L")
        L.save(path_to + "/" + f)


# 第一步，将所有照片转化为灰度图
def turn_images_grep():
    jpg_to_grep(face_path, grep_face_path)
    jpg_to_grep(non_face_path, grep_non_face_path)


# 第二步，提取NPD特征，顺便分好训练集和验证集，保存在文件中
def extract_npd():
    # 人脸
    face_list = np.array(extract_to_list(grep_face_path))
    # 非人脸
    non_face_list = np.array(extract_to_list(grep_non_face_path))
    bound = int(face_list.shape[0]*0.7)
    x_train = np.append(face_list[:bound], (non_face_list[:bound]), axis=0)
    x_valid = np.append(face_list[bound:], (non_face_list[bound:]), axis=0)
    # 保存训练集
    with open(train_npd_path, "wb") as f:
        pickle.dump(x_train, f)
    # 保存测试集
    with open(valid_npd_path, "wb") as f:
        pickle.dump(x_valid, f)


# 将数据集的各个样本的NPD特征以list返回
def extract_to_list(path_from):
    result = []
    for f in os.listdir(path_from):
        i = Image.open(path_from + "/" + f)
        im_array = np.array(i)
        npd = NPDFeature(im_array)
        features = npd.extract()
        result.append(features)
    return result


# 第三步，读取NPD特征
def load_and_split():
    x_train = load_from_file(train_npd_path)
    x_valid = load_from_file(valid_npd_path)

    # print(type(x_train))
    # print(type(x_valid))
    # print(x_train.shape)
    # print(x_valid.shape)

    # y_train = np.array(init_y(bound)).reshape(-1, 1)
    y_train = np.array(init_y(x_train.shape[0]))
    # y_valid = np.array(init_y(len(face_list) - bound)).reshape(-1, 1)
    y_valid = np.array(init_y(x_valid.shape[0]))
    # print(y_train.shape)
    # print(y_valid.shape)
    # print(y_train)
    return x_train, y_train, x_valid, y_valid


def init_y(bound):
    y = []
    for i in range(int(bound)):
        if i < (bound // 2):
            y.append(1)
        else:
            y.append(-1)
    return y


def load_from_file(npd_path):
    result = []
    with open(npd_path, "rb") as npd_file:
        result = pickle.load(npd_file)
    return result


# 第四步，进行训练
def process_boost():
    x_train, y_train, x_valid, y_valid = load_and_split()
    n_weakers_limit = 20
    adaBoost = AdaBoostClassifier(DecisionTreeClassifier, n_weakers_limit)
    adaBoost.fit(x_train, y_train)
    # 测试集预测
    predict_list = adaBoost.predict(x_valid)
    target_names = ['face', 'non_face']
    report = classification_report(y_valid, predict_list, target_names=target_names)
    with open("D:/testing/python/classifier_report.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    # turn_images_grep()
    # extract_npd()
    process_boost()
    pass

