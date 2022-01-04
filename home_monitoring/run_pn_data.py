# posenet
import tensorflow as tf
import cv2
import time
import argparse
import posenet
import numpy as np
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import json
from tensorflow import keras
import copy
from collections import Counter

# firebase
import argparse
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from app.helper.database import FirebaseConnector

# python run_pn_data.py --model 101 --file ./videos_test/test.mp4

# Arguments for program
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

def check_polygon(point, polygon):
    point_result = Point(point)
    polygon_result = Polygon(polygon)
    return polygon_result.contains(point_result)

def read_roi(input_dir):
    with open(input_dir, 'r') as jsonfile:
        config = json.load(jsonfile)
    roi = []
    for i in range(len(config)):
        roi.append(config['xY'+str(i)])
    return roi

def calc_centroid(arr):
    for i in range(1, len(arr)):
        arr[i][1] = arr[i][1] - arr[0][1]
        arr[i][2] = arr[i][2] - arr[0][2]
        arr[i].pop(0)
    arr[0].pop(0)
    arr[0][0], arr[0][1] = 0, 0
    return arr

def calc_acceleration(arr):
    arr_cpy1 = copy.deepcopy(arr)
    arr_cpy2 = copy.deepcopy(arr)
    for i in range(len(arr_cpy1)-1):
        for j in range(len(arr_cpy1[i])):
            arr_cpy1[i+1][j][0] = arr_cpy1[i+1][j][0] - arr_cpy2[i][j][0]
            arr_cpy1[i+1][j][1] = arr_cpy1[i+1][j][1] - arr_cpy2[i][j][1]
    return arr_cpy1

def eval(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
    return precision, recall, f1

def check_state(arr):
    c = Counter(arr)
    val,_ = c.most_common()[0]
    return val

# Initialize fps time
fps_time = 0
roi = read_roi("./json/limit.json")

# Initialize information
batch_size = 16
label_cat = {0:'fall', 1:'sit', 2:'stand', 3:'walk'}
label_cat2 = {'fall':0, 'sit':1, 'stand':2, 'walk':3}
label_count = [0, 0, 0, 0, 0]

def main():
    global fps_time

    with tf.Session() as sess:
        # Load posenet model
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        # Load keras model
        model_keras = keras.models.load_model("./model_keras/gait_keras.h5py")
        # model_keras.summary()
        
        # Check input video
        if args.file is not None:
            cap = cv2.VideoCapture(args.file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        # Initialize writer for output video
        out = cv2.VideoWriter("./output/output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280,720))

        # Initialize firebase
        CREDENTIALS = "./json/cred.json"
        cred = credentials.Certificate(CREDENTIALS)
        firebase_admin.initialize_app(cred, {
            'databaseURL' : 'https://databasedemo-kjre-default-rtdb.firebaseio.com/' # URL of the firebase database
        })
        ref = db.reference("/")
        send_firebase = FirebaseConnector(db,'c')

        main_arr = []
        state_arr = []
        current_state = ""

        while True:
            # Input frame
            try:
                input_image, display_image, output_scale = posenet.read_cap(
                    cap, roi, scale_factor=args.scale_factor, output_stride=output_stride)
            except:
                break

            # Feed frame to program
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            # Perform pose detection
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            # Convert keypoint coordinates to correct scale
            keypoint_coords *= output_scale

            # Draw pose estimation
            overlay_image, pose_coords = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            if len(pose_coords) >= 12:
                pose_coords = pose_coords[:12]

                point1 = [pose_coords[0][1], pose_coords[0][2]]
                point2 = [pose_coords[-1][1], pose_coords[-1][2]]
                if check_polygon(point1, roi) and check_polygon(point2, roi):
                    if len(main_arr) == batch_size:
                        # acceleration calculation for keypoint
                        main_arr_cpy = calc_acceleration(main_arr)
                        # Detect using keras model
                        main_arr_pred = np.array(main_arr_cpy)
                        main_arr_pred = main_arr_pred.reshape(-1, batch_size, 12, 2)
                        main_arr_pred = main_arr_pred.astype('float32')
                        main_arr_pred = main_arr_pred / 255.
                        gait_pred = model_keras.predict(main_arr_pred)
                        if max(gait_pred[0]) > 0.95:
                            cv2.putText(overlay_image, str(label_cat[np.argmax(gait_pred[0])]),(pose_coords[0][1], pose_coords[0][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            label_count[np.argmax(gait_pred[0])] += 1
                            # Send action to firebase if state change
                            if len(state_arr) == 0:
                                send_firebase.addAction("Sochi", str(label_cat[np.argmax(gait_pred[0])]) + "ing")
                                current_state = str(label_cat[np.argmax(gait_pred[0])])
                            else:
                                if len(state_arr) > 14:
                                    checked_state = check_state(state_arr)
                                    if checked_state != current_state:
                                        send_firebase.addAction("Sochi", checked_state + "ing")
                                        current_state = checked_state
                                    state_arr.pop(0)
                            state_arr.append(str(label_cat[np.argmax(gait_pred[0])]))
                        else:
                            cv2.putText(overlay_image, 'unknown',(pose_coords[0][1], pose_coords[0][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # Maintain batch size
                        main_arr.pop(0)
                    main_arr.append(calc_centroid(pose_coords))

            # Show fps
            cv2.putText(overlay_image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            fps_time = time.time()

            # Show result frame
            cv2.imshow('posenet', cv2.resize(overlay_image, (960,540)))
            # overlay_image = cv2.polylines(overlay_image, [np.array(roi)], True, (0,255,0), 2)

            # Write frame to output video
            out.write(overlay_image)
            
            # Stop program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                cap.release()
                out.release()
                break


if __name__ == "__main__":
    option = input("\nDo you want to evaluate? ")

    main()

    if option == "y":
        tp = label_count[label_cat2[args.file[14:-9]]]
        fp = sum(label_count) - label_count[label_cat2[args.file[14:-9]]]
        fn = 0
        precision, recall, f1 = eval(tp, fp, fn)
        print("Precision: {}, Recall: {}, F1: {}".format(precision, recall, f1))