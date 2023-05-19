import numpy as np
from pathlib import Path
import os


class KalmanBoxTracker(object):
    def __init__(self, bbox):
        self.dt = 0.2
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        self.P = np.diag((0.01, 0.01, 0.01, 0.01))
        self.Q = np.eye(self.A.shape[1])
        self.R = np.eye(self.H.shape[0])
        self.x = np.array([bbox[0], bbox[1], 0, 0]).reshape(-1, 1)
        self.id = 0
        self.meas = self.x.copy()

    def predict(self):
        self.x = self.A @ self.x

    def update(self, z):
        self.meas = np.dot(self.H, z)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        y = self.meas - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(S, K.T))

    def get_state(self):
        return self.x.reshape(1, -1)[0]


def filter_boxes(box_list):
    trackers = []
    for box in box_list:
        trackers.append(KalmanBoxTracker(box))

    for i, box in enumerate(box_list):
        trackers[i].predict()

    for i in range(len(trackers)):
        for j in range(i + 1, len(trackers)):
            iou = get_iou(trackers[i].get_state(), trackers[j].get_state())
            if iou > 0.5:
                trackers[i].update(trackers[j].get_state())

    filtered_boxes = []
    for tracker in trackers:
        state = tracker.get_state()
        box = [state[0], state[1], state[0], state[1] + state[3],
               state[0] + state[2], state[1] + state[3], state[0] + state[2], state[1]]
        filtered_boxes.append(box)

    return filtered_boxes


def get_iou(bbox1, bbox2):
    bbox1_c = bbox_transform(bbox1)
    bbox2_c = bbox_transform(bbox2)
    x11, y11, x12, y12 = bbox1_c
    x21, y21, x22, y22 = bbox2_c

    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    bbox1_area = (x12 - x11) * (y12 - y11)
    bbox2_area = (x22 - x21) * (y22 - y21)

    iou = inter_area / (bbox1_area + bbox2_area - inter_area + 1e-7)
    return iou


def bbox_transform(bbox):
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return (x1, y1, x2, y2)


def filter_obj(input_path, output_path):
    vertices = []
    faces = []
    boxes = []
    with open(input_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertex = list(map(float, line.split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = list(map(int, line.split()[1:]))
                face = [i - 1 for i in face]
                faces.append(face)

    for face in faces:
        x1, y1, z1 = np.min(np.array(vertices)[face], axis=0)
        x2, y2, z2 = np.max(np.array(vertices)[face], axis=0)
        box = [x1, y1, x2 - x1, y2 - y1]
        boxes.append(box)

    filtered_boxes = filter_boxes(boxes)

    with open(output_path, 'w') as f:
        for vertex in vertices:
            f.write('v {} {} {}\n'.format(*vertex))
        for i, face in enumerate(faces):
            f.write('f {} {} {}\n'.format(*(np.array(face) + 1)))
            f.write('# Box ({:.2f}, {:.2f}, {:.2f}, {:.2f})\n'.format(*filtered_boxes[i]))


if __name__ == '__main__':
    folder_path = "D:/userdata/桌面/mmdetection3d-master/Out1"
    folder_output = "./Out"
    folders = os.listdir(folder_path)
    NUM = 20
    for i in range(NUM):
        folder_name = folders[i]
        file_name = folder_name + "_pred.obj"
        input_path = folder_path + "/" + folder_name + "/" + file_name
        output_path = folder_output + "/" + folder_name + "_out.obj"
        filter_obj(input_path, output_path)
