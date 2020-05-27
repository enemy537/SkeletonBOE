import pickle
import logging
import math
import cv2
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import pairwise_distances
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True, save_as='name'):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(save_as + ".png", dpi=300, bbox_inches='tight')


class PYPoses:
    def __init__(self):
        self.logger = self.__create_logger__()

        self.points_dic = {0: "Nose", 1: "Neck", 2: "RShoulder", 3: "RElbow", 4: "RWrist", 5: "LShoulder",
                           6: "LElbow", 7: "LWrist", 8: "MidHip", 9: "RHip", 10: "RKnee", 11: "RAnkle",
                           12: "LHip", 13: "LKnee", 14: "LAnkle", 15: "REye", 16: "LEye", 17: "REar",
                           18: "LEar", 19: "LBigToe", 20: "LSmallToe", 21: "LHeel", 22: "RBigToe",
                           23: "RSmallToe", 24: "RHeel", 25: "Background"}

        self.pose_pairs = ((0, 1), (0, 15), (0, 16), (1, 2), (1, 5), (1, 8), (2, 3), (3, 4), (5, 6),
                           (6, 7), (8, 9), (8, 12), (9, 10), (10, 11), (11, 24), (11, 22), (12, 13),
                           (13, 14), (14, 19), (14, 21), (15, 17), (16, 18), (19, 20), (22, 23))

        self.pairs_labels = (0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 1, 1, 3, 3, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1)

        self.joints = {'lshoulder': (1, 5, 6), 'rshoulder': (1, 2, 3), 'lelbow': (5, 6, 7), 'relbow': (2, 3, 4),
                       'lknee': (12, 13, 14), 'rknee': (9, 10, 11), 'lankle': (13, 14, 19), 'rankle': (10, 11, 22),
                       'lpelvis': (10, 9, 8), 'rpelvis': (13, 12, 8)}

    @staticmethod
    def __create_logger__():
        logger = logging.getLogger('TfPoseEstimatorRun')
        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    @staticmethod
    def __length_between_points__(p0, p1):
        return math.hypot(p1[0] - p0[0], p1[1] - p0[1])

    @staticmethod
    def __angle_between_points__(p0, p1, p2):
        a = (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2
        b = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        c = (p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2
        if a * b == 0:
            return -1.0
        return math.acos((a + b - c) / math.sqrt(4 * a * b)) * 180 / math.pi

    def __is_meaningful__(self, points: np.array) -> bool:
        found = False
        for pair in self.pose_pairs:
            partFrom = pair[0]
            partTo = pair[1]
            if points[partFrom] and points[partTo]:
                found = True
        return found

    def __get_angle_point__(self, human, pos):
        pos_list = self.joints[pos]
        pnts = []

        for i in range(3):
            if human[pos_list[i]][2] <= 0.1:
                print('component [%d] incomplete' % (pos_list[i]))
                return pnts

            pnts.append((int(human[pos_list[i]][0]), int(human[pos_list[i]][1])))
        return pnts

    def get_joint_angle(self, human, joint):
        pnts = self.__get_angle_point__(human, joint)
        if len(pnts) != 3:
            self.logger.info('component incomplete')
            return -1

        angle = 0
        if pnts is not None:
            angle = self.__angle_between_points__(pnts[0], pnts[1], pnts[2])
            self.logger.info('{} angle: {}'.format(joint, angle))
        return angle

    def get_points_dic(self):
        return self.points_dic

    def reconstruct_op(self, poseKeypoints, image, skeleton_mode='full', view_mode=True):
        if poseKeypoints.size < 3:
            return False, None
        else:
            skeletons = []
            for person in range(poseKeypoints.shape[0]):
                out = np.zeros_like(image)
                points = []
                for i in range(len(self.points_dic) - 1):
                    point = poseKeypoints[person][i, :2]
                    conf = poseKeypoints[person][i, 2]
                    x = int(point[0])
                    y = int(point[1])

                    # Add a point if it's confidence is higher than threshold.
                    points.append((int(x), int(y)) if conf > 0.3 else None)

                if skeleton_mode == 'partial':
                    partFrom = 1  # Neck
                    partTo = 8  # MidHip
                    if points[partFrom] and points[partTo]:
                        cv2.line(out, points[partFrom], points[partTo], (255, 255, 255), 2)
                        skeletons.append(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) != 0.)

                        if view_mode:
                            cv2.line(image, points[partFrom], points[partTo], (255, 74, 0), 2)
                            cv2.ellipse(image, points[partFrom], (1, 1), 0, 0, 360, (255, 255, 255), cv2.FILLED)
                            cv2.ellipse(image, points[partTo], (1, 1), 0, 0, 360, (255, 255, 255), cv2.FILLED)
                else:
                    for pair in self.pose_pairs:
                        partFrom = pair[0]
                        partTo = pair[1]

                        if points[partFrom] and points[partTo]:
                            cv2.line(out, points[partFrom], points[partTo], (255, 255, 255), 2)
                            if view_mode:
                                cv2.line(image, points[partFrom], points[partTo], (255, 74, 0), 2)
                                cv2.ellipse(image, points[partFrom], (1, 1), 0, 0, 360, (255, 255, 255), cv2.FILLED)
                                cv2.ellipse(image, points[partTo], (1, 1), 0, 0, 360, (255, 255, 255), cv2.FILLED)

                    skeletons.append(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) != 0.)

            if len(skeletons) > 0:
                return True, skeletons
            else:
                return False, None

    def create_graph(self, poseKeypoints):
        if poseKeypoints.size < 3:
            return False, None
        else:
            G = []
            for person in range(poseKeypoints.shape[0]):
                points = []
                for i in range(len(self.points_dic) - 1):
                    point = poseKeypoints[person][i, :2]
                    conf = poseKeypoints[person][i, 2]
                    x = int(point[0])
                    y = int(point[1])

                    # Add a point if it's confidence is higher than threshold.
                    points.append((int(x), int(y)) if conf > 0.3 else None)

                graph = nx.Graph()
                labels = {}

                for pair in self.pose_pairs:
                    partFrom = pair[0]
                    partTo = pair[1]

                    if points[partFrom] and points[partTo]:
                        graph.add_edge(self.points_dic[partFrom], self.points_dic[partTo],
                                       weight=self.__length_between_points__(points[partFrom], points[partTo]))
                        labels[self.points_dic[partFrom]] = partFrom
                        labels[self.points_dic[partTo]] = partTo

                if graph.size() > 0:
                    nx.set_node_attributes(graph, labels, 'label')
                    G.append(graph)

            if len(G) > 0:
                return True, G
            else:
                return False, None

    def empty_dataframe(self):
        columns = ['{}-{}'.format(self.points_dic[p1], self.points_dic[p2]) for p1, p2 in self.pose_pairs]
        [columns.append(j) for j in self.joints.keys()]
        columns.append('label')
        return pd.DataFrame(columns=columns)

    def large_empty_dataframe(self):
        point_comb = list(itertools.combinations(range(25), 2))
        joint_comb = list(itertools.combinations(range(25), 3))
        points = ['{}-{}'.format(self.points_dic[p1], self.points_dic[p2]) for p1, p2 in point_comb]
        joints = ['{}-{}-{}'.format(self.points_dic[p1], self.points_dic[p2], self.points_dic[p3]) for p1, p2, p3 in
                  joint_comb]
        columns = points + joints
        columns.append('label')
        return pd.DataFrame(columns=columns)

    def create_pandas_frames(self, poseKeypoints, df):
        if poseKeypoints.size < 3:
            return False, None
        else:
            valid = 0
            for person in range(poseKeypoints.shape[0]):
                points = []
                for i in range(len(self.points_dic) - 1):
                    point = poseKeypoints[person][i, :2]
                    conf = poseKeypoints[person][i, 2]
                    x = int(point[0])
                    y = int(point[1])

                    # Add a point if it's confidence is higher than threshold.
                    points.append((int(x), int(y)) if conf > 0.3 else None)

                sk_dict = dict(zip(df.columns, np.full(len(df.columns), np.nan)))
                found = False

                for pair in self.pose_pairs:
                    partFrom = pair[0]
                    partTo = pair[1]

                    if points[partFrom] and points[partTo]:
                        key = '{}-{}'.format(self.points_dic[partFrom], self.points_dic[partTo])
                        val = self.__length_between_points__(points[partFrom], points[partTo])
                        sk_dict[key] = val
                        found = True

                for key, val in self.joints.items():
                    p0, p1, p2 = val
                    if points[p0] and points[p1] and points[p2]:
                        ang = self.__angle_between_points__(points[p0], points[p1], points[p2])
                        sk_dict[key] = ang

                if found:
                    df.loc[len(df)] = sk_dict
                    valid += 1

            if valid > 0:
                return True, valid
            else:
                return False, None

    def create_large_pandas_frames(self, poseKeypoints, df, metric):
        if poseKeypoints.size < 3:
            return False, None
        else:
            valid = 0
            for person in range(poseKeypoints.shape[0]):
                points = []
                for i in range(len(self.points_dic) - 1):
                    point = poseKeypoints[person][i, :2]
                    conf = poseKeypoints[person][i, 2]
                    x = int(point[0])
                    y = int(point[1])

                    # Add a point if it's confidence is higher than threshold.
                    points.append((int(x), int(y)) if conf > 0.3 else None)

                sk_dict = dict(zip(df.columns, np.full(len(df.columns), np.nan)))
                found = False

                if self.__is_meaningful__(points):
                    point_comb = list(itertools.combinations(range(25), 2))
                    joint_comb = list(itertools.combinations(range(25), 3))

                    for pair in point_comb:
                        partFrom = pair[0]
                        partTo = pair[1]

                        if points[partFrom] and points[partTo]:
                            key = '{}-{}'.format(self.points_dic[partFrom], self.points_dic[partTo])
                            # val = self.__length_between_points__(points[partFrom], points[partTo])
                            p1, p2 = np.asarray(points[partFrom]).reshape(1, 2), np.asarray(points[partTo]).reshape(1,
                                                                                                                    2)
                            val = pairwise_distances(p1, p2, metric=metric).mean()
                            sk_dict[key] = val
                            found = True

                    for val in joint_comb:
                        p0, p1, p2 = val
                        if points[p0] and points[p1] and points[p2]:
                            ang = self.__angle_between_points__(points[p0], points[p1], points[p2])
                            key = '{}-{}-{}'.format(self.points_dic[p0], self.points_dic[p1], self.points_dic[p2])
                            sk_dict[key] = ang

                if found:
                    df.loc[len(df)] = sk_dict
                    valid += 1

            if valid > 0:
                return True, valid
            else:
                return False, None


def label_conversor(label, sk_num):
    if type(label) == list:
        if len(label) == sk_num:
            return label
        elif len(label) == 1:
            return np.ones(sk_num, dtype=int) * label[0]
    elif type(label) == tuple:
        slh_common, slh_diff = label
        rt = np.ones(sk_num, dtype=int) * slh_common[0]
        rt[slh_diff[0]] = slh_diff[1]
        return rt


def create_dataset(graph_dic, op_dic, op, metric):
    yx = []
    df = op.large_empty_dataframe()

    for k in graph_dic.keys():
        _, num = op.create_large_pandas_frames(op_dic[k], df, metric)
        labels = label_conversor(graph_dic[k], num)
        for l in labels:
            yx.append(l)

    df['label'] = yx
    return df


if __name__ == '__main__':
    op_dic = pickle.load(open('db_skeleton', 'rb'))
    graph_dic = pickle.load(open('annotation_dic', 'rb'))

    op = PYPoses()
    eval_metric = ["auc", "error"]
    classifier = xgb.XGBClassifier(silent=False,
                                   scale_pos_weight=1,
                                   learning_rate=0.1,
                                   colsample_bytree=0.8,
                                   subsample=0.8,
                                   objective='binary:logistic',
                                   n_estimators=1000,
                                   reg_alpha=0.3,
                                   max_depth=5,
                                   gamma=0.1)

    metrics = ['euclidean', 'haversine', 'canberra', 'braycurtis']

    for metric in metrics:
        print('Creating dataset for {} metric'.format(metric))
        df = create_dataset(graph_dic, op_dic, op, metric)
        dataset = df.sample(frac=1, random_state=123)
        X, y = dataset.iloc[:, :-1], dataset.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        print('Now training...')
        classifier.fit(X_train, y_train, eval_metric=eval_metric, verbose=True)
        print('Training finished.')
        y_pred = classifier.predict(X_test)

        report = 'Results for {} \n'.format(metric)
        report += classification_report(y_test, y_pred, digits=3)
        report += '\n \n'

        with open('out.txt', 'a+') as f:
            for line in report.split('\n'):
                f.write(line + '\n')
            f.close()

        plot_confusion_matrix(cm=confusion_matrix(y_test, y_pred),
                              normalize=True,
                              target_names=[1, 2, 3, 4, 5, 6, 7, 8],
                              title='Confusion Matrix',
                              save_as=metric + '_matrix')

        plt.tight_layout()
        plt.figure(figsize=(10, 6))
        feat_imp = pd.Series(classifier.feature_importances_, index=X_test.columns).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.savefig(metric + "_feature_score.png", dpi=300, bbox_inches='tight')
