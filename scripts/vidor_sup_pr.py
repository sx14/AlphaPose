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


def load_tid2cls(trajs):
    tid2cls = {}
    for tid, traj in enumerate(trajs):
        tid2cls[tid] = traj['category']
    return tid2cls


def load_frame_dets(trajs):

    def get_video_len(trajs):
        return max([int(traj['end_fid']) + 1 for traj in trajs])

    video_len = get_video_len(trajs)
    frame_dets = [[] for _ in range(video_len)]
    for tid, traj in enumerate(trajs):
        fid2box = traj['trajectory']
        for fid in sorted(fid2box):
            box = fid2box[fid]
            frame_dets[int(fid)].append({
                'tid': tid,
                'bbox': {'xmin': box[0], 'ymin': box[1],
                          'xmax': box[2], 'ymax': box[3]}})
    return frame_dets


def add_pose(fid2dets, vid_trajs, human_cates):

    for traj in vid_trajs:
        if traj['category'] in human_cates:
            traj['pose'] = {}

    for fid in range(len(fid2dets)):
        frm_dets = fid2dets[fid]
        for det in frm_dets:
            if 'kps' in det:
                kps = det['kps']
                tid = det['tid']
                vid_trajs[tid]['pose']['%06d' % fid] = kps
    return


split = ('val', 'validation')
data_root = '../data/vidor_hoid_mini'
pose_root = os.path.join(data_root, 'Pose', 'VID', split[0])
traj_path = os.path.join(data_root, 'object_trajectories_%s_det.json' % split[0])
traj_with_pose_path = os.path.join(data_root, 'object_trajectories_%s_det_with_pose.json' % split[0])

# load trajectory detections
with open(traj_path) as f:
    traj_file = json.load(f)
    all_trajs = traj_file['results']

human_cates = {'adult', 'child', 'baby'}
for pid_vid in sorted(all_trajs):

    pkg_id, vid_id = pid_vid.split('/')
    vid_pose_file_path = os.path.join(pose_root, pkg_id, vid_id+'.json')
    with open(vid_pose_file_path) as f:
        fid2pose = load_pose(json.load(f))

    vid_trajs = all_trajs[pid_vid]
    tid2cate = load_tid2cls(vid_trajs)
    fid2dets = load_frame_dets(vid_trajs)

    human_box_cnt = 0
    human_box_pose_cnt = 0
    for fid in range(len(fid2dets)):
        frm_dets = fid2dets[fid]
        frm_kps = fid2pose[fid]
        frm_pboxes = [get_skeleton_box(kps) for kps in frm_kps]
        for det_idx in range(len(frm_dets)):
            tid = frm_dets[det_idx]['tid']

            if tid2cate[tid] not in human_cates:
                continue

            human_box_cnt += 1
            xmin = frm_dets[det_idx]['bbox']['xmin']
            xmax = frm_dets[det_idx]['bbox']['xmax']
            ymin = frm_dets[det_idx]['bbox']['ymin']
            ymax = frm_dets[det_idx]['bbox']['ymax']
            tbox = [xmin, ymin, xmax, ymax]

            max_iou = 0
            max_ind = 0
            for kps_ind, pbox in enumerate(frm_pboxes):
                curr_iou = iou(pbox, tbox)
                if curr_iou > max_iou:
                    max_iou = curr_iou
                    max_ind = kps_ind

            if max_iou > 0.4:
                human_box_pose_cnt += 1
                frm_dets[det_idx]['kps'] = frm_kps[max_ind]
            else:
                frm_dets[det_idx]['kps'] = None

        if human_box_cnt > 0:
            print('[%s/%s]: %.2f' % (pkg_id, vid_id, human_box_pose_cnt * 1.0 / human_box_cnt))
        else:
            print('[%s/%s]: %.2f' % (pkg_id, vid_id, -1))

    add_pose(fid2dets, vid_trajs, human_cates)

with open(traj_with_pose_path, 'w') as f:
    json.dump(traj_file, f)
print('%s saved' % traj_with_pose_path)