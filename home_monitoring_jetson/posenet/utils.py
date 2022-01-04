import cv2
import numpy as np
import copy
import posenet.constants


def valid_resolution(width, height, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    return target_width, target_height


def mask(img, roi):
    contours = np.array(roi)
    mask = np.zeros(img.shape, dtype=np.uint8)
    cv2.fillPoly(mask, pts=[contours], color=(255,255,255))
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def _process_input_vid(source_img, roi, scale_factor=1.0, output_stride=16):
    source_img = cv2.resize(source_img, (1280,720))
    masked_img = mask(source_img, roi)

    target_width, target_height = valid_resolution(
        masked_img.shape[1] * scale_factor, masked_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([masked_img.shape[0] / target_height, masked_img.shape[1] / target_width])

    input_img = cv2.resize(masked_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.reshape(1, target_height, target_width, 3)
    return input_img, source_img, scale


def _process_input_img(source_img, scale_factor=1.0, output_stride=16):
    source_img = cv2.resize(source_img, (1280,720))

    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])

    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input_img = input_img * (2.0 / 255.0) - 1.0
    input_img = input_img.reshape(1, target_height, target_width, 3)
    return input_img, source_img, scale


def read_cap(cap, roi, scale_factor=1.0, output_stride=16):
    res, img = cap.read()
    if not res:
        raise IOError("webcam failure")
    return _process_input_vid(img, roi, scale_factor, output_stride)


def read_imgfile(path, scale_factor=1.0, output_stride=16):
    img = cv2.imread(path)
    return _process_input_img(img, scale_factor, output_stride)


def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = []
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        results.append(
            np.array([[int(keypoint_coords[left][::-1][0]), int(keypoint_coords[left][::-1][1])], 
                    [int(keypoint_coords[right][::-1][0]), int(keypoint_coords[right][::-1][1])]]).astype(np.int32),
        )
    return results


def draw_skel_and_kp_img(
    img, instance_scores, keypoint_scores, keypoint_coords,
    min_pose_score=0.5, min_part_score=0.5):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []

    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)

        adjacent_keypoints.extend(new_keypoints)

        for ki, (ks, kc) in enumerate(zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :])):
            if ks < min_part_score:
                continue
            face_pose = [0,1,2,3,4]
            if ki not in face_pose:
                cv_keypoints.append([ki, int(kc[1]), int(kc[0])])

    for i in range(len(cv_keypoints)):
        # cv2.putText(out_img, str(cv_keypoints[i][0]), (cv_keypoints[i][1], cv_keypoints[i][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        if cv_keypoints[i][1] < 0:
            cv_keypoints[i][1] = 0
        if cv_keypoints[i][2] < 0:
            cv_keypoints[i][2] = 0
        #out_img = cv2.circle(out_img, (cv_keypoints[i][1], cv_keypoints[i][2]), 1, (0, 255, 0), 1)
    
    # Draw lines
    #out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    
    return out_img, cv_keypoints


def centeroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def get_bbox(arr):
    arr_cpy = arr[:]

    arr_cpy.sort(key=lambda x:x[1])
    x, w = arr_cpy[0][1], arr_cpy[-1][1]
    arr_cpy.sort(key=lambda x:x[2])
    y, h = arr_cpy[0][2], arr_cpy[-1][2]

    center = centeroid(np.array([[x,y],[w,h]]))

    y = y - ((h-y)*0.325)
    h = h + (((h-y)*0.325)/2)

    x = int(center[0] - ((abs(h-y)*(1/2))/2))
    w = x + int(abs(h-y)*(1/2))
    y = int(center[1] - (abs(h-y)/1.75))
    h = y + int(abs(h-y))
    
    return x, y, w, h


def draw_skel_and_kp(
    img, instance_scores, keypoint_scores, keypoint_coords,
    min_pose_score=0.5, min_part_score=0.5):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints_check = []
    cv_keypoints = []
    index_five = -1
    index_sixteen = -1

    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)

        adjacent_keypoints.extend(new_keypoints)

        for ki, (ks, kc) in enumerate(zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :])):
            if ks < min_part_score:
                continue
            face_pose = [0,1,2,3,4]
            if ki not in face_pose:
                cv_keypoints_check.append([ki, int(kc[1]), int(kc[0])])
                if ki == 5:
                    index_five = len(cv_keypoints_check)-1
                elif ki == 16:
                    index_sixteen = len(cv_keypoints_check)-1

                if index_five != -1 and index_sixteen != -1:
                    if index_sixteen - index_five == 11:
                        for i in range(index_five, index_sixteen+1):
                            cv_keypoints.append(cv_keypoints_check[i])
                        for i in range(0, len(cv_keypoints)-11, 12):
                            x, y, w, h = get_bbox(cv_keypoints[i:i+11])
                            if h - y > 125:
                                out_img = cv2.rectangle(out_img, (x,y), (w,h), (255,0,255), 1, 1)
                            else:
                                cv_keypoints = cv_keypoints[i+11:]
                    index_five = -1
                    index_sixteen = -1

    cv2.putText(out_img, "Detected " + str(int(len(cv_keypoints)/12)), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    for i in range(len(cv_keypoints)):
        # cv2.putText(out_img, str(cv_keypoints[i][0]), (cv_keypoints[i][1], cv_keypoints[i][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        if cv_keypoints[i][1] < 0:
            cv_keypoints[i][1] = 0
        if cv_keypoints[i][2] < 0:
            cv_keypoints[i][2] = 0
        out_img = cv2.circle(out_img, (cv_keypoints[i][1], cv_keypoints[i][2]), 1, (0, 255, 0), 1)
    
    # Draw lines
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    
    return out_img, cv_keypoints