import cv2
import tensorflow as tf
import numpy as np
import mediapipe as mp
from pyfirmata import Arduino, util, SERVO
import random
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
cTime = 0
pTime = 0

#Arduino Variables
board = Arduino('COM3')
iterator = util.Iterator(board)
iterator.start()

LED_Rock = board.get_pin('d:9:o')
LED_Paper = board.get_pin('d:10:o')
LED_Scissors = board.get_pin('d:11:o')
LED_Win = board.get_pin('d:5:o')
LED_Lose = board.get_pin('d:7:o')
LED_Draw = board.get_pin('d:6:o')

Play_Button = board.get_pin('d:2:i')
Off_Button = board.get_pin('d:4:i')

Servo = board.get_pin('d:3:s')

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm )
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy)
                # if id == 0:
                #     cv2.circle(img,(cx,cy),15,(255,67,45),cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 255), 2)

    cv2.imshow("img", img)

    Servo.write(60)
    if Play_Button.read() == 1:
        for i in range(4):
            for i in range(60, 0, -1):
                Servo.write(i)
            time.sleep(0.25)
            for i in range(0, 60, 1):
                Servo.write(i)
            time.sleep(0.25)
        cv2.imwrite("User_Choice.png", img)
        break
cap.release()
cv2.destroyAllWindows()

#NN Predictor
nnin = cv2.imread('Stock Images/scissors.jpg')
imgResize = cv2.resize(nnin, (224, 224))
imgResize = imgResize[np.newaxis, :]

model = tf.keras.models.load_model("PF.model")

prediction = model.predict(x=imgResize, steps=1, verbose=0)
print(np.argmax(prediction))

list = [0, 1, 2]
Comp_Choice = random.choice(list)
User_Choice = random.choice(list)

if (User_Choice == 0) & (Comp_Choice == 0):
    LED_Rock.write(1)
    time.sleep(1.0)
    LED_Draw.write(1)
elif (User_Choice == 0) & (Comp_Choice == 1):
    LED_Paper.write(1)
    time.sleep(1.0)
    LED_Lose.write(1)
elif (User_Choice == 0) & (Comp_Choice == 2):
    LED_Scissors.write(1)
    time.sleep(1.0)
    LED_Win.write(1)

elif (User_Choice == 1) & (Comp_Choice == 0):
    LED_Rock.write(1)
    time.sleep(1.0)
    LED_Win.write(1)
elif (User_Choice == 1) & (Comp_Choice == 1):
    LED_Paper.write(1)
    time.sleep(1.0)
    LED_Draw.write(1)
elif (User_Choice == 1) & (Comp_Choice == 2):
    LED_Scissors.write(1)
    time.sleep(1.0)
    LED_Lose.write(1)

elif (User_Choice == 2) & (Comp_Choice == 0):
    LED_Rock.write(1)
    time.sleep(1.0)
    LED_Lose.write(1)
elif (User_Choice == 2) & (Comp_Choice == 1):
    LED_Paper.write(1)
    time.sleep(1.0)
    LED_Win.write(1)
elif (User_Choice == 2) & (Comp_Choice == 2):
    LED_Scissors.write(1)
    time.sleep(1.0)
    LED_Draw.write(1)

print("You Chose: ", User_Choice)
print("Computer Chose: ", Comp_Choice)
board.exit()