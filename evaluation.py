import os
import json
import numpy as np
# import the necessary libraries
import cv2
import pycocotools.mask as mask_util
from pycocotools.coco import COCO


def write_json(all_results, outputpath, for_eval=False, outputfile='alphapose-results.json'):
    '''
    all_result: result dict of predictions
    outputpath: output directory
    '''
    json_results = []
    for im_res in all_results:
        im_name = im_res['imgname']
        for human in im_res['result']:
            keypoints = []
            result = {}
            if for_eval:
                result['image_id'] = int(os.path.basename(im_name).split('.')[0].split('_')[-1])
            else:
                result['image_id'] = os.path.basename(im_name)
            result['category_id'] = 1

            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            result['keypoints'] = keypoints
            result['score'] = float(pro_scores)
            if 'box' in human.keys():
                result['box'] = human['box']
            #pose track results by PoseFlow
            if 'idx' in human.keys():
                result['idx'] = human['idx']
            
            # 3d pose
            if 'pred_xyz_jts' in human.keys():
                pred_xyz_jts = human['pred_xyz_jts']
                pred_xyz_jts = pred_xyz_jts.cpu().numpy().tolist()
                result['pred_xyz_jts'] = pred_xyz_jts

            json_results.append(result)

    with open(os.path.join(outputpath, outputfile), 'w') as json_file:
        json_file.write(json.dumps(json_results))

def eval(gt_keypoints, pred_keypoints):
    results = {
        'AP': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'total_gt': 0,
        'total_pred': 0,
        'correct_pred': 0
    }


    # get the ground truth keypoints for this image
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)
    gt_keypoints = []
    for annotation in annotations:
        # get the keypoints for this annotation
        keypoints = annotation['keypoints']
        # reshape the keypoints into a 2D array
        keypoints = np.array(keypoints).reshape(-1, 3)
        # only keep the keypoints that are visible (i.e. have a confidence value of 2)
        keypoints = keypoints[keypoints[:, 2] == 2]
        gt_keypoints.append(keypoints)

    # get the image file path
    image_info = coco.loadImgs(image_id)[0]
    image_path = 'path/to/coco/images/' + image_info['file_name']
    # load the image
    image = cv2.imread(image_path)

    # run the keypoint detection model on the image
    predicted_keypoints = keypoint_detection_model(image)

    # iterate over the predicted keypoints
    for predicted_keypoints_for_instance in predicted_keypoints:
        # compute the IoU (Intersection over Union) between the predicted keypoints and the ground truth keypoints
        ious = mask_util.iou(predicted_keypoints_for_instance, gt_keypoints, np.zeros((len(gt_keypoints),)))
        # find the index of the
