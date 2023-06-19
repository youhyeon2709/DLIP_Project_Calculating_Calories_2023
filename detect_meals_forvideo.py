import numpy as np
import cv2 
from matplotlib import pyplot as plt
import time
import torch
from pathlib import Path
from PIL import Image
import pandas as pd

# Load YOLOv5 model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='.\\best.pt')

# Initialize the # of Frame
frame_count = 0

# Initialize the start time
start_time = 0

# Load DLIP_calories.csv 
df = pd.read_csv('DLIP_calories.csv')

#Initialize color
sky_blue = (0,191,255)
lime_green = (277,255,0)
white = (255,255,255)
green = (206, 250, 5)

total_caloreis = 0
total_carbohydrates = 0
total_protein = 0
total_fat = 0

input_path = 'Evaluation_MS_S&T.mp4'

# file video open
vid = cv2.VideoCapture(input_path)
ret, frame = vid.read()

#height, width, channels = frame.shape
height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
width  = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

# Output AVI file settings
output_path = 'Detect_Evaluation_MS_S&T.mp4'  
output_fps = vid.get(cv2.CAP_PROP_FPS)
output_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_path, fourcc, output_fps, output_size)

# fuction of precessing detection object
def process_results(results, image):

    global total_caloreis 
    global total_carbohydrates 
    global total_protein 
    global total_fat 
    global output_txt_path

    total_caloreis = 0
    total_carbohydrates = 0
    total_protein = 0
    total_fat = 0
    output_txt_path = 0

    one_food_info =""
    total_food_info = ""
    deteion_food_name =""

    infor_height = 60

    # get results from objects
    objects = results.pandas().xyxy[0]
    # print objects information
    print(objects)

    cv2.putText(image, f"Frame: {frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, sky_blue, 2)

    # draw information of object in image 
    for _, obj in objects.iterrows():
        label = obj['name']
        confidence = obj['confidence']
        x_min = int(obj['xmin'])
        y_min = int(obj['ymin'])
        x_max = int(obj['xmax'])
        y_max = int(obj['ymax'])

        if label in df['food name'].values:
            food_data = df[df['food name'] == label].iloc[0]
            caloreis = food_data['energy(kcal)']
            carbohydrates = food_data['carbohydrates(g)']
            protein = food_data['protein(g)']
            fat = food_data['fat(g)']

            total_caloreis += caloreis
            total_carbohydrates += carbohydrates
            total_protein += protein
            total_fat += fat

            deteion_food_name += f"{label} "

        one_food_info = f"Food: {label} Cal: {caloreis:.1f} kcal Carbs: {carbohydrates:.1f} g Pro: {protein:.1f} g Fat: {fat:.1f} g"
        cv2.putText(image, one_food_info, (10, infor_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 2)
        infor_height += 20

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), lime_green, 2)
        cv2.putText(image, f'{label}: {confidence:.1f}', (x_min, y_min - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, lime_green, 2)

    #output_file.write(f"{frame_count } {deteion_food_name}\n") 
    total_food_info = f"Carbs: {total_carbohydrates:.1f}/246 g Pro: {total_protein:.1f}/67 g Fat: {total_fat:.1f}/60 g"

    cv2.putText(image, f"Total | Cal: {total_caloreis:.1f}/2100 kcal", (10, int(height*0.9)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, green, 2)
    cv2.putText(image, total_food_info, (10, int(height*0.9)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)

    return image

        


'''
MAIN CODE 
'''


while True:
    ret, frame = vid.read()

    if not ret:
        print('video in not open')
        break

    print(f'height:{height}\n')
    print(f'width:{width}\n')

    # Increase frame count
    frame_count += 1

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    model.conf = 0.1
    results = model(frame) 

    # processing result
    frame = process_results(results, frame)

    # image show
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #resized_frame = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Result', frame)

    # Adding frames to the output video
    output_video.write(frame)

    #Exit if this is the last frame.
    if vid.get(cv2.CAP_PROP_POS_FRAMES) == vid.get(cv2.CAP_PROP_FRAME_COUNT):
        break

    if cv2.waitKey(1) == ord('q'):
        break
    
vid.release()
output_video.release()
cv2.destroyAllWindows()

