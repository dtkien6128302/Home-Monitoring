
import tensorflow as tf
import cv2
import time
import argparse
import posenet
import numpy as np
import glob
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import json
from pathlib import Path

# python get_img_data_multi.py --model 101 --file ./videos/*.mp4

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

# Initialize fps time
fps_time = 0
roi = read_roi("./json/limit.json")

def main(file):
    global fps_time

    with tf.Session() as sess:
        # Load posenet model
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']
        
        # Check input video
        if file is not None:
            cap = cv2.VideoCapture(file)
        else:
            cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        # Initialize writer for output video
        out = cv2.VideoWriter("./output/" + file[9:-4] + "_collect_output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, (1280,720))

        file_name_output = 0

        # Create folder to save cropped images
        Path("./images_output/" + str(file[9:-4])).mkdir(parents=True, exist_ok=True)

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
            overlay_image, pose_coords = posenet.draw_skel_and_kp_img(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            if len(pose_coords) >= 12:
                pose_coords = pose_coords[:12]

                point1 = [pose_coords[0][1], pose_coords[0][2]]
                point2 = [pose_coords[-1][1], pose_coords[-1][2]]
                if check_polygon(point1, roi) and check_polygon(point2, roi):
                    # Draw Bbox
                    x, y, w, h = get_bbox(pose_coords)
                    conv_img = cv2.resize(overlay_image[y:h, x:w], (180, 360), interpolation = cv2.INTER_NEAREST)
                    cv2.imshow("img", conv_img)
                    cv2.imwrite("./images_output/" + str(file[9:-4]) + "/" + str(file_name_output) + ".png", conv_img)
                    file_name_output += 1
                    overlay_image = cv2.rectangle(overlay_image, (x,y), (w,h), (255,0,255), 1, 1)

            # Show fps
            cv2.putText(overlay_image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            fps_time = time.time()

            # Show result frame
            # overlay_image = cv2.polylines(overlay_image, [np.array(roi)], True, (0,255,0), 2)
            cv2.imshow('posenet', cv2.resize(overlay_image, (960,540)))

            # Write frame to output video
            out.write(overlay_image)
            
            # Stop program
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                cap.release()
                out.release()
                break

if __name__ == "__main__":
    for file in glob.glob(args.file):
        main(file)