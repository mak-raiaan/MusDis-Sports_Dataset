import csv
import math
import os

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

from reba_scoring import groupA
from reba_scoring import groupB
from reba_scoring import groupC


def store_coordinates_csv(image_name, keypoints, csv_path):
    """append one image's 66 pixel coordinates to a CSV file.

    Args:
        image_name: output image filename.
        keypoints: iterable containing 33 ``(x, y)`` points.
        csv_path: destination coordinate CSV path.
    """
    coordinates = [value for point in keypoints for value in point]
    if len(coordinates) != 66:
        raise ValueError("expected exactly 33 two-dimensional keypoints")
    with open(csv_path, mode="a", newline="") as file:
        csv.writer(file).writerow([image_name, *coordinates])

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

# Function to score based on angles
def score_trunk(angle):
    if 0 <= angle <= 5:
        return 1
    elif 5 < angle <= 20:
        return 2
    elif 20 < angle <= 60:
        return 3
    elif angle > 60:
        return 4

def score_neck(angle):
    if 0 <= angle <= 20:
        return 1
    else:
        return 2

def score_leg(angle):
    if 30 <= angle <= 60:
        return 1
    else:
        return 2

def score_upper_arm(angle):
    if -20 <= angle <= 20:
        return 1
    elif 20 < angle <= 45 or -20 > angle >= -45:
        return 2
    elif 45 < angle <= 90 or -45 > angle >= -90:
        return 3
    else:
        return 4

def score_lower_arm(angle):
    if 60 <= angle <= 100:
        return 1
    else:
        return 2

def score_wrist(angle):
    if -15 <= angle <= 15:
        return 1
    else:
        return 2

# Function to assign REBA score based on key point angles
def assign_reba_score(keypoints):
    l_eye = keypoints[mp_pose.PoseLandmark.LEFT_EYE.value]
    r_eye = keypoints[mp_pose.PoseLandmark.RIGHT_EYE.value]
    nose = keypoints[mp_pose.PoseLandmark.NOSE.value]
    l_shoulder = keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    l_hip = keypoints[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_hip = keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value]
    l_ear = keypoints[mp_pose.PoseLandmark.LEFT_EAR.value]
    r_ear = keypoints[mp_pose.PoseLandmark.RIGHT_EAR.value]
    l_knee = keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value]
    l_ankle = keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    r_knee = keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    r_ankle = keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    l_elbow = keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    r_elbow = keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    l_wrist = keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value]
    r_wrist = keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    l_index = keypoints[mp_pose.PoseLandmark.LEFT_INDEX.value]
    r_index = keypoints[mp_pose.PoseLandmark.RIGHT_INDEX.value]

    #for neck
    neck_mid_point = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
    distance_bwteen_nose_mid_neck = math.sqrt((nose[0] - neck_mid_point[0])**2 + (nose[1] - neck_mid_point[1])**2)
    #for trunk
    hip_mid_point = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
    distance_bwteen_neck_mid_point_hip_mid_point = math.sqrt((neck_mid_point[0] - hip_mid_point[0])**2 + (neck_mid_point[1] - hip_mid_point[1])**2)
    

    # Calculate angles
    trunk_angle = calculate_angle((hip_mid_point[0],hip_mid_point[1]- distance_bwteen_neck_mid_point_hip_mid_point), hip_mid_point, neck_mid_point)
    neck_angle = calculate_angle(nose , (neck_mid_point[0],neck_mid_point[1]-distance_bwteen_nose_mid_neck), (nose[0] ,neck_mid_point[1]-distance_bwteen_nose_mid_neck))
    l_leg_angle = 180 - calculate_angle(l_hip, l_knee, l_ankle)
    r_leg_angle = 180 - calculate_angle(r_hip, r_knee, r_ankle)
    l_upper_arm_angle = calculate_angle(l_elbow, l_shoulder, (l_shoulder[0], l_hip[1]))
    r_upper_arm_angle = calculate_angle(r_elbow, r_shoulder, (r_shoulder[0], r_hip[1]))
    l_lower_arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
    r_lower_arm_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)
    l_wrist_angle = 180 - calculate_angle(l_elbow, l_wrist, l_index)
    r_wrist_angle = 180 - calculate_angle(r_elbow, r_wrist, r_index)

    # Get scores
    trunk_score = score_trunk(trunk_angle)
    neck_score = score_neck(neck_angle)
    l_leg_score = score_leg(l_leg_angle)
    r_leg_score = score_leg(r_leg_angle)
    l_upper_arm_score = score_upper_arm(l_upper_arm_angle)
    r_upper_arm_score = score_upper_arm(r_upper_arm_angle)
    l_lower_arm_score = score_lower_arm(l_lower_arm_angle)
    r_lower_arm_score = score_lower_arm(r_lower_arm_angle)
    l_wrist_score = score_wrist(l_wrist_angle)
    r_wrist_score = score_wrist(r_wrist_angle)
    
    # Combine scores to get final REBA score
    groupA_score = groupA(neck_score, trunk_score, r_leg_score)
    groupB_score = groupB(r_lower_arm_score, r_wrist_score, r_upper_arm_score)

    print(groupA_score)
    print(groupB_score)

    reba_score = (
        groupC(groupA_score, groupB_score)
    )

    return reba_score

# Function to process and annotate image
def process_and_annotate_image(image_path, output_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        keypoints = []
        height, width, _ = image.shape
        for landmark in results.pose_landmarks.landmark:
            if landmark.visibility < 0.5:
                return -1, -1, []
            keypoints.append(
                (
                    int(landmark.x * width),
                    int(landmark.y * height),
                )
            )

        reba_score = assign_reba_score(keypoints)
        
        # Annotate image with REBA score
        font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(image, f"REBA Score: {reba_score + 1}", (30, 70), font, 2, (0, 255, 0), 5, cv2.LINE_AA)

        # Draw keypoints and connections
        for idx, keypoint in enumerate(keypoints):
            cv2.circle(image, keypoint, 5, (0, 255, 0), -1)
            cv2.putText(image, str(idx), keypoint, font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        # Optionally, draw connections
        connections = [
            (mp_pose.PoseLandmark.RIGHT_SHOULDER.value, mp_pose.PoseLandmark.RIGHT_ELBOW.value),
            (mp_pose.PoseLandmark.RIGHT_ELBOW.value, mp_pose.PoseLandmark.RIGHT_WRIST.value),
            (mp_pose.PoseLandmark.LEFT_SHOULDER.value, mp_pose.PoseLandmark.LEFT_ELBOW.value),
            (mp_pose.PoseLandmark.LEFT_ELBOW.value, mp_pose.PoseLandmark.LEFT_WRIST.value),
            (mp_pose.PoseLandmark.LEFT_HIP.value, mp_pose.PoseLandmark.LEFT_KNEE.value),
            (mp_pose.PoseLandmark.LEFT_KNEE.value, mp_pose.PoseLandmark.LEFT_ANKLE.value),
            (mp_pose.PoseLandmark.RIGHT_HIP.value, mp_pose.PoseLandmark.RIGHT_KNEE.value),
            (mp_pose.PoseLandmark.RIGHT_KNEE.value, mp_pose.PoseLandmark.RIGHT_ANKLE.value)
        ]
        
        for start_idx, end_idx in connections:
            cv2.line(image, keypoints[start_idx], keypoints[end_idx], (255, 0, 0), 2)

        # Save the annotated image
        cv2.imwrite(output_path, image)


        # create_structure = generate_structure(image_path,output_path)
        return reba_score, 1, keypoints
    else:
        return -1,-1, []
        print("No pose detected.", image_path)

# File name for the CSV
csv_file = "image_coordinates.csv"

# Writing to the CSV
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    l = [str(i) for i in range(1, 67)]
    l.insert(0, "Image")
    print(l)
    writer.writerow(l)



# Function to process images in a folder
def process_images(input_folder, output_folder, csv_path):
    os.makedirs(output_folder, exist_ok=True)
    
    # List to hold data for the CSV file
    data = []

    index = 0
    for idx, filename in enumerate(os.listdir(input_folder)):
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Check for both png and jpg (including jpeg) files

            # Define the new filename
            new_filename = f"image{index + 1}.png"  # You can keep the output as .png or match the input extension
            index+=1
            # Paths for input and output images
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, new_filename)
            
            # Process the image and get the REBA score (Assuming a function 'assign_reba_score')
            reba_score, create_structure, keypoints = process_and_annotate_image(input_path, output_path)
            
            if reba_score != -1 and create_structure != -1 and len(keypoints) !=0:
                # Append the new filename and REBA score to the data list
                data.append([new_filename, reba_score])
                store_coordinates_csv(new_filename, keypoints, csv_file)
            else:
                index-=1
                print(f"No pose detected in {filename}, skipping...")


    df = pd.DataFrame(data, columns=["images", "reba_score"])
    print(df)
    df.to_csv(csv_path, index=False)

    print(f"CSV file created at {csv_path}")

# Specify input and output folders
input_folder = 'E:/REBA Risk Prediction/images for new dataset/dataset3'
output_folder = 'E:/REBA Risk Prediction/output_for_new_dataset/dataset with human image and key points'
csv_path = 'E:/REBA Risk Prediction/output_for_new_dataset/labels_image_with_key_points.csv'

# Process the images and create the CSV
process_images(input_folder, output_folder, csv_path)

# Release MediaPipe Pose
pose.close()