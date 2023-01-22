
import argparse
import shutil
import pandas as pd
import numpy as np 
import cv2
import os
import random

CAR_CONFIDENCE_THRESH = 70.0
PERSON_CONFIDENCE_THRESH = 50.0

def generate_neighboring_orientations(current_orientation):
    items = current_orientation.split('-')
    pan = int(items[0])
    zoom = int(items[-1])
    if pan == 0:
        left_horz = 330
    else:
        left_horz = int(items[0]) - 30
    if pan == 330:
        right_horz = 0
    else:
        right_horz = int(items[0]) + 30

    if len(items) == 4:
        tilt = int(items[2]) * -1
    else:
        tilt = int(items[1])
    top_tilt = tilt + 15
    bottom_tilt = tilt - 15

    if tilt == 30:
        return [ f'{left_horz}-{tilt}-{zoom}', 
                 f'{right_horz}-{tilt}-{zoom}', 
                 current_orientation, 
                 f'{right_horz}-{bottom_tilt}-{zoom}', 
                 f'{pan}-{bottom_tilt}-{zoom}',
                 f'{left_horz}-{bottom_tilt}-{zoom}',
                ]
    elif tilt == -30:
        return [ 
                 f'{left_horz}-{tilt}-{zoom}', 
                 f'{right_horz}-{tilt}-{zoom}', 
                 current_orientation, 
                 f'{right_horz}-{top_tilt}-{zoom}', 
                 f'{pan}-{top_tilt}-{zoom}',
                 f'{left_horz}-{top_tilt}-{zoom}',
                ]

    return [ 
             f'{left_horz}-{top_tilt}-{zoom}',
             f'{left_horz}-{tilt}-{zoom}', 
             f'{left_horz}-{bottom_tilt}-{zoom}', 
             f'{right_horz}-{top_tilt}-{zoom}', 
             f'{right_horz}-{tilt}-{zoom}', 
             f'{right_horz}-{bottom_tilt}-{zoom}', 
             current_orientation, 
             f'{pan}-{top_tilt}-{zoom}', 
             f'{pan}-{bottom_tilt}-{zoom}' ]

def write_to_file_with_df(f, image_file, orientation_df, orientation, object_type):
    if object_type != 'car' and object_type != 'person':
        raise Exception('Incorrect object type')
    count = 0
    for idx, row in orientation_df.iterrows():
        if object_type == 'car':
            if row['class'] == 'car' and row['confidence'] >= CAR_CONFIDENCE_THRESH:
                xmin = row['left']
                xmax = row['right']
                ymin = row['top']
                ymax = row['bottom']
                f.write(f'{image_file},1280,720,car,{xmin},{ymin},{xmax},{ymax}\n')
        if object_type == 'person':
            if row['class'] == 'person' and row['confidence'] >= PERSON_CONFIDENCE_THRESH:
                xmin = row['left']
                xmax = row['right']
                ymin = row['top']
                ymax = row['bottom']
                f.write(f'{image_file},1280,720,person,{xmin},{ymin},{xmax},{ymax}\n')
        

ap = argparse.ArgumentParser()
ap.add_argument('rectlinear', help='Directory of raw frames')
ap.add_argument('inference_dir', help='Directory of inference results (include model folder)')
ap.add_argument('fixed_orientations_file', help='CSV with best fixed orientations')
ap.add_argument('frame_begin', type=int, help='Beginning frame num')
ap.add_argument('frame_limit', type=int, help='Ending frame num')
ap.add_argument('objecttype', type=str, help='car or person')
ap.add_argument('outfile',  type=str, help='Output file')

ap.add_argument('--ignore-begin', default=0, type=int, help='Beginnign frame num to ignore')  
ap.add_argument('--ignore-limit', default=0, type=int, help='Ending frame num to ignore')  
ap.add_argument("--per-orientation", action="store_true") 
args = ap.parse_args()


object_type = args.objecttype

best_fixed_df = pd.read_csv(args.fixed_orientations_file)

frame_limit_to_orientation = {}
for idx, row in best_fixed_df.iterrows():
    if object_type == 'car' and row['class'] == 'car':
        if row['frame_limit'] not in frame_limit_to_orientation:
            frame_limit_to_orientation[row['frame_limit']] = []  
        if row['orientation'] not in frame_limit_to_orientation[row['frame_limit']]:
            frame_limit_to_orientation[row['frame_limit']].append(row['orientation'])
    elif object_type == 'person' and row['class'] == 'person':
        if row['frame_limit'] not in frame_limit_to_orientation:
            frame_limit_to_orientation[row['frame_limit']] = []
        if row['orientation'] not in frame_limit_to_orientation[row['frame_limit']]:
            frame_limit_to_orientation[row['frame_limit']].append( row['orientation'])
current_frame = args.frame_begin
frame_bounds = [(1,1161), (1162,1663), (1664,2823), (2824,3966), (3967, 4983), (4984, 6075), (6076, 7194),  (7195, 7920), (16939,18418)]
current_frame = args.frame_begin
frames_added = []
orientation_to_avg_count_list = {}

result_idx = -1
prev_result_idx = -1

print('Generating daataset for ', args.outfile)
with open(args.outfile, 'w') as f:
    f.write('filename,width,height,class,xmin,ymin,xmax,ymax\n')
    while current_frame <= args.frame_limit:
        if current_frame <= frame_bounds[0][1]:
            result_idx = 0
        elif current_frame <= frame_bounds[1][1]:
            result_idx = 1
        elif current_frame <= frame_bounds[2][1]:
            result_idx = 2
        elif current_frame <= frame_bounds[3][1]:
            result_idx = 3
        elif current_frame <= frame_bounds[4][1]:
            result_idx = 4
        elif current_frame <= frame_bounds[5][1]:
            result_idx = 5
        elif current_frame <= frame_bounds[6][1]:
            result_idx = 6
        elif current_frame <= frame_bounds[7][1]:
            result_idx = 7
        elif current_frame <= frame_bounds[8][1]:
            result_idx = 8
        else:
            result_idx = 0 
        sub_frame_begin = frame_bounds[result_idx][0]
        sub_frame_limit = frame_bounds[result_idx][1]
        orientations = frame_limit_to_orientation[sub_frame_limit][:1]

        if current_frame % 6 != 0:
            current_frame += 1
            continue
        if current_frame >= args.ignore_begin and current_frame <= args.ignore_limit:
            current_frame += 1
            continue
        #######
        # For training on first 20% of frames
        total_frames = int((frame_bounds[result_idx][1] - frame_bounds[result_idx][0]) * 0.4)
        if current_frame >= frame_bounds[result_idx][0] + total_frames:
            current_frame += 1
            continue
        ######
#        if current_frame % 2 != 0:
#            current_frame += 1
#            continue
        ######
        # FOr training with 66% (dispersed) of training set
#        frames_added.append(current_frame)
#        if len(frames_added) >= 3:
#            current_frame += 1
#            frames_added.clear()
#            continue
#        ######


#        if int(orientations[0][:orientations[0].index('-')]) % 60 != 0:
#            current_frame += 1
#            continue
        all_orientations = []
        for current_orientation in orientations:
            neighboring_orientations = generate_neighboring_orientations(current_orientation)
            for no in neighboring_orientations:
                if no not in all_orientations:
                    all_orientations.append(no)
        print('current orienttion ', current_orientation)
        print('all orientations ', all_orientations)
        for o in all_orientations:
            neighbor_result_orientation_dir = os.path.join(args.inference_dir, o)
            inference_file = os.path.join(neighbor_result_orientation_dir, f'frame{current_frame}.csv')
            if os.path.getsize(inference_file) > 0:
                orientation_df = pd.read_csv(inference_file)
                orientation_df.columns = ['left', 'top', 'right', 'bottom', 'class', 'confidence']
                orig_image_file = os.path.join(args.rectlinear, o, f'frame{current_frame}.jpg')

                image_file = f'{o}-frame{current_frame}.jpg'
                dest = f'images/test/{image_file}'
                shutil.copy(orig_image_file, dest)
                write_to_file_with_df(f, image_file, orientation_df, o, object_type)

        current_frame += 1


