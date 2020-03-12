import os
import cv2
import json
import numpy as np

color_h = (0, 0, 255)
color_o = (240, 176, 0)
color_point = (0, 255, 255)
color_bone = (176, 240, 0)
color_part = (0, 190, 255)
color_grey = (207, 207, 207)

ho_thick = 10
kp_thick = 6
part_thick = 6
bone_thick = 4


key_points = ["nose",
              "left_eye", "right_eye",
              "left_ear", "right_ear",
              "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow",
              "left_wrist", "right_wrist",
              "left_hip", "right_hip",
              "left_knee", "right_knee",
              "left_ankle", "right_ankle"]

key_point_connections = [
    [0, 1],     # nose->left eye
    [0, 2],     # nose->right eye
    [1, 3],     # left eye->left ear
    [2, 4],     # right eye->right ear
    [0, 5],     # nose->left shoulder
    [0, 6],     # nose->right shoulder
    [5, 7],     # left_shoulder->left elbow
    [6, 8],     # right_shoulder->right elbow
    [7, 9],     # left elbow->left wrist
    [8, 10],    # right elbow->right wrist
    [5, 11],    # left shoulder->left hip
    [6, 12],    # right shoulder->right hip
    [11, 13],   # left hip->left knee
    [12, 14],   # right hip->right knee
    [13, 15],   # left knee->left ankle
    [14, 16],   # right knee->right ankle
    [5, 6],     # left shoulder->right shoulder
    [11, 12],   # left hip->right hip
]


def draw_human_object_skeleton(hbox, skeleton, im):
    # figure2: human-skeleton        (human)
    # figure2: human-object-skeleton (full image)
    # figure4: human-object-skeleton (union box)

    if isinstance(im, str):
        im = cv2.imread(im)
    cv2.rectangle(im, (hbox[0], hbox[1]), (hbox[2], hbox[3]), color_h, thickness=ho_thick)
    kps = []
    for i in range(skeleton.shape[0]):
        raw_kp = skeleton[i, :].astype(np.int).tolist()
        kps.append((raw_kp[0], raw_kp[1]))

    for kp in kps:
        cv2.circle(im, kp, kp_thick, color_point, -1)

    for kp_cnts in key_point_connections:
        cv2.line(im, kps[kp_cnts[0]], kps[kp_cnts[1]], color_bone, bone_thick)

    return im


def draw_box(box, im):
    if isinstance(im, str):
        im = cv2.imread(im)
    cv2.rectangle(im, (box[0], box[1]), (box[2], box[3]), color_h, thickness=ho_thick)
    return im


if __name__ == '__main__':
    anno_root = '../data/vidor_hoid_mini/anno_with_pose/validation'
    data_root = '../data/vidor_hoid_mini/Data/VID/val'

    pkg_id = '0000'
    vid_id = '2401075277'

    anno_path = os.path.join(anno_root, pkg_id, vid_id+'.json')
    data_path = os.path.join(data_root, pkg_id, vid_id)

    with open(anno_path) as f:
        anno = json.load(f)

    trajs = anno['trajectories']
    for frm_idx in range(len(trajs)):
        frm_path = os.path.join(data_path, '%06d.JPEG' % frm_idx)
        im = cv2.imread(frm_path)
        frm_dets = trajs[frm_idx]
        for det in frm_dets:
            box = [det['bbox']['xmin'], det['bbox']['ymin'],
                   det['bbox']['xmax'], det['bbox']['ymax']]
            if 'kps' in det and det['kps'] is not None:
                kps = np.array(det['kps']).reshape((17, 3))
                draw_human_object_skeleton(box, kps, im)
            else:
                draw_box(box, im)

        cv2.imshow('123', im)
        cv2.waitKey(0)

