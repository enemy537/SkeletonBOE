import cv2
from abc import ABC, abstractmethod
import os
import sys
import pickle


class DEEParch(ABC):

    @abstractmethod
    def detect(self, data, out_mode='image'):
        pass

    @abstractmethod
    def to_string(self):
        return str()

class OpenPOSE(DEEParch):
    def __init__(self):
        os.path.dirname(os.path.realpath(__file__))
        sys.path.append('python')

        from openpose import pyopenpose as op

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)

        params = dict()
        params["model_folder"] = "models/"
        params["num_gpu"] = 1
        params["num_gpu_start"] = 0

        # Starting OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def __get_datum__(self):
        from openpose import pyopenpose as op
        return op.Datum()

    def detect(self, image, out_mode='image'):
        # Image processor
        datum = self.__get_datum__()
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop([datum])
        skeleton = datum.cvOutputData
        points = datum.poseKeypoints

        if out_mode != 'image':
            return True, points
        else:
            return True, skeleton

    def to_string(self):
        return 'op'

if __name__ == '__main__':
    op = OpenPOSE()
    raw = pickle.load(open('packd.txt', 'rb'), encoding='latin1')

    img = raw[i][0]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img[::-1]

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(50)