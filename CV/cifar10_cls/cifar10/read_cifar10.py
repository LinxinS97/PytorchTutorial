import os
import cv2
import numpy as np


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


label_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

import glob

train_list = glob.glob("F:\OneDrive\WeakSupervision\PytorchTutorial\CV\cifar10_cls\cifar10\data_batch_*")
test_list = glob.glob("F:\OneDrive\WeakSupervision\PytorchTutorial\CV\cifar10_cls\cifar10\\test_batch")

save_path = "F:\OneDrive\WeakSupervision\PytorchTutorial\CV\cifar10_cls\cifar10\TRAIN"
save_path_test = "F:\OneDrive\WeakSupervision\PytorchTutorial\CV\cifar10_cls\cifar10\\TEST"

# for l in train_list:
for l in test_list:
    print(l)
    l_dict = unpickle(l)
    print(l_dict.keys())

    for im_idx, im_data in enumerate(l_dict[b'data']):

        im_label = l_dict[b'labels'][im_idx]
        im_name = l_dict[b'filenames'][im_idx]

        # print(im_idx)
        # print(im_data)
        # print(im_label)
        # print(im_name)

        im_label_name = label_name[im_label]
        im_data = np.reshape(im_data, [3, 32, 32])  # reshape没办法改变数据内存的排列顺序，所以得先用32*32*3来读取
        im_data = np.transpose(im_data, (1, 2, 0))  # 将channel放到最后

        # cv2.imshow("im_data", cv2.resize(im_data, (200, 200)))
        # cv2.waitKey(0)

        # TRAIN
        # if not os.path.exists("{}/{}".format(save_path, im_label_name)):
        #     os.mkdir("{}/{}".format(save_path, im_label_name))
        #
        # cv2.imwrite("{}/{}/{}".format(save_path, im_label_name, im_name.decode('utf-8')), im_data)

        # TEST
        if not os.path.exists("{}/{}".format(save_path_test, im_label_name)):
            os.mkdir("{}/{}".format(save_path_test, im_label_name))

        cv2.imwrite("{}/{}/{}".format(save_path_test, im_label_name, im_name.decode('utf-8')), im_data)

