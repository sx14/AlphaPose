import os
import json
from collections import defaultdict


def iou(box1, box2):
    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2

    xmini = max(xmin1, xmin2)
    ymini = max(ymin1, ymin2)
    xmaxi = min(xmax1, xmax2)
    ymaxi = min(ymax1, ymax2)

    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    areai = max((xmaxi - xmini + 1), 0) * max((ymaxi - ymini + 1), 0)

    return (areai) * 1.0 / (area1 + area2 - areai)


def load_pose(res_pose):
    fid2pose = defaultdict(list)
    for pose_inst in res_pose:
        fid = pose_inst['image_id'].split('.')[0]
        kps = pose_inst['keypoints']
        fid2pose[int(fid)].append(kps)
    return fid2pose


def get_skeleton_box(kps):

    xmin = 9999
    ymin = 9999
    xmax = -999
    ymax = -999

    assert len(kps) == 51
    for j in range(0, len(kps), 3):
        x = kps[j+0]
        y = kps[j+1]

        xmin = min(xmin, x)
        ymin = min(ymin, y)
        xmax = max(xmax, x)
        ymax = max(ymax, y)

    return [xmin, ymin, xmax, ymax]


def load_tid2cls(res_traj):
    traj_info_list = res_traj['subject/objects']
    tid2cls = {}
    for traj_info in traj_info_list:
        tid2cls[traj_info['tid']] = traj_info['category']
    return tid2cls


split = ('val', 'validation')
data_root = '../data/vidor_hoid_mini'
pose_root = os.path.join(data_root, 'Pose', 'VID', split[0])
anno_root = os.path.join(data_root, 'anno', split[1])
anno_pose_root = os.path.join(data_root, 'anno_with_pose', split[1])
if not os.path.exists(anno_pose_root):
    os.makedirs(anno_pose_root)

human_cates = {'adult', 'child', 'baby'}
for pkg_id in os.listdir(pose_root):
    pkg_root = os.path.join(pose_root, pkg_id)
    for vid_id in os.listdir(pkg_root):
        vid_pose_file_path = os.path.join(pkg_root, vid_id)
        vid_anno_file_path = os.path.join(anno_root, pkg_id, vid_id)

        with open(vid_pose_file_path) as f:
            res_pose = json.load(f)
            fid2pose = load_pose(res_pose)
        with open(vid_anno_file_path) as f:
            res_traj = json.load(f)
            tid2cate = load_tid2cls(res_traj)

        human_box_cnt = 0
        human_box_pose_cnt = 0
        trajs = res_traj['trajectories']
        for fid in range(len(trajs)):
            traj_boxes = trajs[fid]
            frm_kps = fid2pose[fid]
            frm_pboxes = [get_skeleton_box(kps) for kps in frm_kps]
            for box_idx in range(len(traj_boxes)):
                tid = traj_boxes[box_idx]['tid']

                if tid2cate[tid] not in human_cates:
                    continue

                human_box_cnt += 1
                xmin = traj_boxes[box_idx]['bbox']['xmin']
                xmax = traj_boxes[box_idx]['bbox']['xmax']
                ymin = traj_boxes[box_idx]['bbox']['ymin']
                ymax = traj_boxes[box_idx]['bbox']['ymax']
                tbox = [xmin, ymin, xmax, ymax]

                max_iou = 0
                max_ind = 0
                for kps_ind, kps in enumerate(frm_kps):
                    pbox = frm_pboxes[kps_ind]
                    curr_iou = iou(pbox, tbox)
                    if curr_iou > max_iou:
                        max_iou = curr_iou
                        max_ind = kps_ind

                if max_iou > 0.4:
                    human_box_pose_cnt += 1
                    traj_boxes[box_idx]['kps'] = frm_kps[max_ind]
                else:
                    traj_boxes[box_idx]['kps'] = None
        print('[%s/%s]: %.2f' % (pkg_id, vid_id.split('.')[0],
                                 human_box_pose_cnt * 1.0 / human_box_cnt))

        vid_anno_with_pose_file_root = os.path.join(anno_pose_root, pkg_id)
        if not os.path.exists(vid_anno_with_pose_file_root):
            os.makedirs(vid_anno_with_pose_file_root)
        vid_anno_with_pose_file_path = os.path.join(vid_anno_with_pose_file_root, vid_id)
        with open(vid_anno_with_pose_file_path, 'w') as f:
            json.dump(res_traj, f)




