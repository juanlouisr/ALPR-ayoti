import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow import keras


img = cv.imread(r'test images/B3023KEZ.jpg')
img = cv.resize(img, (1024,768))
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(20,20))
img_opening = cv.morphologyEx(img_gray, cv.MORPH_OPEN, kernel)
img_norm = img_gray - img_opening
(thresh, img_norm_bw) = cv.threshold(img_norm, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)


contours_vehicle, hierarchy = cv.findContours(img_norm_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
index_plate_candidate = []
index_counter_contour_vehicle = 0
for contour_vehicle in contours_vehicle:
    x,y,w,h = cv.boundingRect(contour_vehicle)
    aspect_ratio = w/h
    if w >= 200 and aspect_ratio <= 4 : 
        
        index_plate_candidate.append(index_counter_contour_vehicle)
    
    index_counter_contour_vehicle += 1
img_show_plate = img.copy() 
img_show_plate_bw = cv.cvtColor(img_norm_bw, cv.COLOR_GRAY2RGB)

if len(index_plate_candidate) == 0:
    print("Plat nomor tidak ditemukan")

elif len(index_plate_candidate) == 1:
    print("ditemukan 1")
    x_plate,y_plate,w_plate,h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[0]])
    
    cv.rectangle(img_show_plate,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)

    cv.rectangle(img_show_plate_bw,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)
    img_plate_gray = img_gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]
else:
    print('Dapat dua lokasi plat, pilih lokasi plat kedua')
    x_plate,y_plate,w_plate,h_plate = cv.boundingRect(contours_vehicle[index_plate_candidate[1]])
    cv.rectangle(img_show_plate,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)
    cv.rectangle(img_show_plate_bw,(x_plate,y_plate),(x_plate+w_plate,y_plate+h_plate),(0,255,0),5)
    img_plate_gray = img_gray[y_plate:y_plate+h_plate, x_plate:x_plate+w_plate]


(thresh, img_plate_bw) = cv.threshold(img_plate_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3,3))

img_plate_bw = cv.morphologyEx(img_plate_bw, cv.MORPH_OPEN, kernel)
contours_plate, hierarchy = cv.findContours(img_plate_bw, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 

index_chars_candidate = []

index_counter_contour_plate = 0 #idx

img_plate_rgb = cv.cvtColor(img_plate_gray,cv.COLOR_GRAY2BGR)
img_plate_bw_rgb = cv.cvtColor(img_plate_bw, cv.COLOR_GRAY2RGB)

for contour_plate in contours_plate:
    x_char,y_char,w_char,h_char = cv.boundingRect(contour_plate)

    if h_char >= 40 and h_char <= 60 and w_char >=10:
        index_chars_candidate.append(index_counter_contour_plate)
        cv.rectangle(img_plate_rgb,(x_char,y_char),(x_char+w_char,y_char+h_char),(0,255,0),5)
        cv.rectangle(img_plate_bw_rgb,(x_char,y_char),(x_char+w_char,y_char+h_char),(0,255,0),5)

    index_counter_contour_plate += 1


if index_chars_candidate == []:

    print('Karakter tidak tersegmentasi')
else:

    score_chars_candidate = np.zeros(len(index_chars_candidate))
    counter_index_chars_candidate = 0
    for chars_candidateA in index_chars_candidate:
        xA,yA,wA,hA = cv.boundingRect(contours_plate[chars_candidateA])
        for chars_candidateB in index_chars_candidate:

            if chars_candidateA == chars_candidateB:
                continue
            else:
                xB,yB,wB,hB = cv.boundingRect(contours_plate[chars_candidateB])
                y_difference = abs(yA - yB)
                if y_difference < 11:
                    
                    score_chars_candidate[counter_index_chars_candidate] = score_chars_candidate[counter_index_chars_candidate] + 1 

        counter_index_chars_candidate += 1

    index_chars = []
    chars_counter = 0

    for score in score_chars_candidate:
        if score == max(score_chars_candidate):
            index_chars.append(index_chars_candidate[chars_counter])
        chars_counter += 1

    img_plate_rgb2 = cv.cvtColor(img_plate_gray, cv.COLOR_GRAY2BGR)
    for char in index_chars:
        x, y, w, h = cv.boundingRect(contours_plate[char])
        cv.rectangle(img_plate_rgb2,(x,y),(x+w,y+h),(0,255,0),5)
        cv.putText(img_plate_rgb2, str(index_chars.index(char)),(x, y + h + 50), cv.FONT_ITALIC, 2.0, (0,0,255), 3)
    x_coors = []

    for char in index_chars:
        x, y, w, h = cv.boundingRect(contours_plate[char])
        x_coors.append(x)
    x_coors = sorted(x_coors)
    index_chars_sorted = []
    for x_coor in x_coors:
        for char in index_chars:
            x, y, w, h = cv.boundingRect(contours_plate[char])
            if x_coors[x_coors.index(x_coor)] == x:
                index_chars_sorted.append(char)

    img_plate_rgb3 = cv.cvtColor(img_plate_gray, cv.COLOR_GRAY2BGR)
    for char_sorted in index_chars_sorted:
        x,y,w,h = cv.boundingRect(contours_plate[char_sorted])
        cv.rectangle(img_plate_rgb3,(x,y),(x+w,y+h),(0,255,0),5)
        cv.putText(img_plate_rgb3, str(index_chars_sorted.index(char_sorted)),(x, y + h + 50), cv.FONT_ITALIC, 2.0, (0,0,255), 3)


    img_height = 40 
    img_width = 40

    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    model = keras.models.load_model('my_model')
    num_plate = []

    for char_sorted in index_chars_sorted:
        x,y,w,h = cv.boundingRect(contours_plate[char_sorted])
        char_crop = cv.cvtColor(img_plate_bw[y:y+h,x:x+w], cv.COLOR_GRAY2BGR)
        char_crop = cv.resize(char_crop, (img_width, img_height))
        img_array = keras.preprocessing.image.img_to_array(char_crop)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0]) 

        num_plate.append(class_names[np.argmax(score)])
    
    plate_number = ''.join(num_plate)
    
    print(plate_number)

