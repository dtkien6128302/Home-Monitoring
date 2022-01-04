import os
import cv2
import time 
from datetime import datetime
import copy
import json
import numpy as np
import argparse
import posenet
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from flask import Flask, render_template, Response
import tensorflow as tf
from tensorflow import keras

# FUNCTIONS

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

# ARGUMENTS

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()

# VARIABLES
app = Flask(__name__)

# Initialize fps time
fps_time = 0
roi = read_roi("./json/limit.json")

# Initialize information
batch_size = 16
label_cat = {0:'fall', 1:'sit', 2:'stand', 3:'walk'}
label_cat2 = {'fall':0, 'sit':1, 'stand':2, 'walk':3}
label_count = [0, 0, 0, 0, 0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/liveview')
def liveview():
    return render_template('liveview.html')

def generate_frame():
    global fps_time
    
    # Initialize information
    batch_size = 16
    label_cat = {0:'fall', 1:'sit', 2:'stand', 3:'walk'}
    label_cat2 = {'fall':0, 'sit':1, 'stand':2, 'walk':3}
    label_count = [0, 0, 0, 0, 0]
    
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

        main_arr = []

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
                        else:
                            cv2.putText(overlay_image, 'unknown',(pose_coords[0][1], pose_coords[0][2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        # Maintain batch size
                        main_arr.pop(0)
                    main_arr.append(calc_centroid(pose_coords))

            # Show fps
            cv2.putText(overlay_image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            fps_time = time.time()
            ret, buffer = cv2.imencode('.jpg',overlay_image)
            frame = buffer.tobytes()
            
            yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        cap.release()
        cv2.destroyAllWindows()

@app.route('/video')
def video():
    return Response(generate_frame(),mimetype="multipart/x-mixed-replace; boundary=frame")
            
if __name__ == '__main__':
    app.run(host= '10.10.2.59', debug=True)
    print(datetime.now())
    
    # python main.py --model 101 --file rtsp://admin:cctv12345@10.10.2.24:554/Streaming/Channels/1101
