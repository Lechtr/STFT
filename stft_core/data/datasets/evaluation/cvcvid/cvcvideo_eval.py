import os
import numpy as np
from stft_core.structures.boxlist_ops import boxlist_iou

# TODO: eval anpassen auf mehrere Klassen, aktuell wird einfach nur geschaut ob Polyp gefunden wurde, also ungleich Klasse 0 ist
def cvcvideo_detection_eval(pred_boxlists, gt_boxlists, score_thrs):
    assert len(gt_boxlists) == len(pred_boxlists), "Length of gt and pred lists need to be same."

    detection_tp = np.zeros((len(pred_boxlists), score_thrs.shape[0]), dtype=np.int)
    detection_fp = np.zeros((len(pred_boxlists), score_thrs.shape[0]), dtype=np.int)
    detection_tn = np.zeros((len(pred_boxlists), score_thrs.shape[0]), dtype=np.int)
    detection_fn = np.zeros((len(pred_boxlists), score_thrs.shape[0]), dtype=np.int)
    pos_num=0
    neg_num=0

    # for each eval Image one jeweils (len(521))
    # gt_boxlists: [BoxList(num_boxes=1, image_width=384, image_height=288, mode=xyxy), ..., BoxList(num_boxes=1, image_width=384, image_height=288, mode=xyxy)]
    # pred_boxlists: [BoxList(num_boxes=300, image_width=384, image_height=288, mode=xyxy), BoxList(num_boxes=300, image_width=384, image_height=288, mode=xyxy), ...
    # score_thrs: array([0.6])
    for idx, (gt_boxlist, pred_boxlist) in enumerate(zip(gt_boxlists, pred_boxlists)):
        # gt_boxlist: BoxList(num_boxes=1, image_width=384, image_height=288, mode=xyxy)
        # pred_boxlist: BoxList(num_boxes=300, image_width=384, image_height=288, mode=xyxy)
        gt_bbox = gt_boxlist.bbox.numpy() # array([[177., 181., 262., 241.]], dtype=float32)
        gt_label = gt_boxlist.get_field("labels").numpy() # array([2])
        # no Polyp in the image
        # if gt_label.sum()==0: # 2
        #     have_obj=False
        #     neg_num+=1
        # else:
        #     have_obj=True
        #     pos_num+=1

        # pred_bbox = pred_boxlist.bbox.numpy() # array([[ 29.61034 ,  64.35797 , 265.34225 , 252.00955 ],..., [164.34525 , 181.94551 , 208.72359 , 226.4958  ]], dtype=float32)
        pred_score = pred_boxlist.get_field("scores").numpy() # array([0.32293808, 0.31038857, ..., 0.12737879], dtype=float32)
        pred_label = pred_boxlist.get_field("labels").numpy() # array([1, 2, 1, 2, 1, ....])

        for score_idx, score_thr in enumerate(score_thrs):
            det_inds = pred_score >= score_thr # array([False, ..., False])
            highscore_score = pred_score[det_inds]
            # highscore_bbox = pred_bbox[det_inds]
            highscore_label = pred_label[det_inds]

            # für Klassifiaktion die Scores ignorieren und Annahme, dass immer nur 1 Polyp pro Bild
            # da bereits nach Score sortiert dann einfach das erste Label nehmen
            # für hp vs ad
            # TP = gt ad, pred ad
            # FP = gt hp, pred ad
            # TN = gt hp, pred hp
            # FN = gt ad, pred hp
            for polyp_in_image_index, gt in enumerate(gt_label):
                # hp == 1; ad == 2
                if gt == 2 and pred_label[polyp_in_image_index] == 2:
                    detection_tp[idx, score_idx] += 1
                elif gt == 1 and pred_label[polyp_in_image_index] == 2:
                    detection_fp[idx, score_idx] += 1
                elif gt == 1 and pred_label[polyp_in_image_index] == 1:
                    detection_tn[idx, score_idx] += 1
                elif gt == 2 and pred_label[polyp_in_image_index] == 1:
                    detection_fn[idx, score_idx] += 1


            # Boxen predicted
            # if highscore_bbox.shape[0]>0:
            #     if have_obj:
            #         detection_tp[idx,score_idx]+=1 # array([[0], [0],....])
            #     else:
            #         detection_fp[idx,score_idx]+=1
            # # keine Boxen predicted
            # else:
            #     if have_obj:
            #         detection_fn[idx,score_idx]+=1
            #     else:
            #         detection_tn[idx,score_idx]+=1



    TP = np.sum(detection_tp, axis=0)
    FP = np.sum(detection_fp, axis=0)
    TN = np.sum(detection_tn, axis=0)
    FN = np.sum(detection_fn, axis=0)
    det_pos = TP + FP
    det_neg = TN + FN


    # das passt so, für Multi-class muss die Berechnung der TP,FP,... angepasst werden
    Precision = 100*TP/(TP+FP+1e-7)
    Recall = 100*TP/(TP+FN+1e-7)
    Accuracy = 100*(TP+TN)/(TP+TN+FP+FN+1e-7)
    Specificity = 100*TN/(TN+FP+1e-7)
    F1_score = 2*Precision*Recall/(Precision+Recall+1e-7)
    F2_score = 5*Precision*Recall/(4*Precision + Recall+1e-7)

    return np.vstack((Precision, Recall, Accuracy, Specificity, F1_score, F2_score)), \
        detection_tp, detection_fp, detection_tn, detection_fn



#det_bbox: x1,y1,x2,y2
#gt_box: x1,y1,x2,y2
def polyp_center(det_box, gt_box):

    center_x_det = float(det_box[0] + (det_box[2]-det_box[0])/2)
    center_y_det = float(det_box[1] + (det_box[3]-det_box[1])/2)

    if center_x_det > gt_box[0] and center_x_det < gt_box[2]:
        if center_y_det > gt_box[1] and center_y_det < gt_box[3]:
            return True
        else:
            return False
    return False



def cvcvideo_localization_center_eval(pred_boxlists, gt_boxlists, score_thrs):
    assert len(gt_boxlists) == len(pred_boxlists), "Length of gt and pred lists need to be same."

    detection_tp = np.zeros((len(pred_boxlists), score_thrs.shape[0]), dtype=np.int)
    detection_fp = np.zeros((len(pred_boxlists), score_thrs.shape[0]), dtype=np.int)
    detection_fn = np.zeros((len(pred_boxlists), score_thrs.shape[0]), dtype=np.int)
    gt_num = 0

    for idx, (gt_boxlist, pred_boxlist) in enumerate(zip(gt_boxlists, pred_boxlists)):

        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()
        if gt_label.sum()==0:
            continue
        gt_num+=gt_label.sum()

        pred_bbox = pred_boxlist.bbox.numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()

        for score_idx, score_thr in enumerate(score_thrs):
            det_inds = pred_score >= score_thr
            highscore_score = pred_score[det_inds]
            highscore_bbox = pred_bbox[det_inds]

            if highscore_bbox.shape[0]>0:
                #have bbox with higher score
                center_flag_list = []
                for temp_bbox in highscore_bbox:
                    center_flags = []
                    for gt_idx in range(gt_label.sum()):
                        center_flag = polyp_center(temp_bbox, gt_bbox[gt_idx])
                        center_flags.append(center_flag)
                    center_flag_list.append(center_flags)
                center_flag_list = np.array(center_flag_list) #Mpred*Ngt

                det_for_each_gt = center_flag_list.sum(0)
                for gt_idx in range(gt_label.sum()):
                    if det_for_each_gt[gt_idx] > 0:
                        detection_tp[idx,score_idx] += 1
                    else:
                        detection_fn[idx,score_idx] += 1
                detection_fp[idx,score_idx] += max((highscore_bbox.shape[0] - center_flag_list.sum()),0)
            else:
                #bbox score lower than score_thr
                detection_fn[idx,score_idx] += gt_label.sum()

    TP = np.sum(detection_tp, axis=0)
    FP = np.sum(detection_fp, axis=0)
    FN = np.sum(detection_fn, axis=0)

    Precision = 100*TP/(TP+FP+1e-7)
    Recall = 100*TP/(TP+FN+1e-7)
    F1_score = 2*Precision*Recall/(Precision+Recall+1e-7)
    F2_score = 5*Precision*Recall/(4*Precision + Recall+1e-7)
   
    return np.vstack((Precision, Recall, F1_score, F2_score)), \
        detection_tp, detection_fp, detection_fn