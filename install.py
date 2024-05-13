import cv2
import mediapipe
import pyttsx3
import math

camera = cv2.VideoCapture(0)
engine = pyttsx3.init()
mpHands = mediapipe.solutions.hands
hands = mpHands.Hands()
mpDraw = mediapipe.solutions.drawing_utils

while True:

    success, img = camera.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hlms = hands.process(imgRGB)
    height, width, channel = img.shape
    results = hands.process(imgRGB)

    print(hlms.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # İşaret parmağı 6. 7. ve 8. landmarklar arası açıları
            # İşaret parmağının eklemlerinin indeksleri (MediaPipe Hands kütüphanesine göre)
            index_base = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_MCP]
            index_mid = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_PIP]
            index_tip = hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]

            # İşaret parmağının eklemlerinin koordinatlarını alın
            index_base_coords = (index_base.x, index_base.y)
            index_mid_coords = (index_mid.x, index_mid.y)
            index_tip_coords = (index_tip.x, index_tip.y)

            # İki eklemin koordinatları arasındaki mesafeyi ve açıyı hesapla
            index_distance_base_mid = math.dist(index_base_coords, index_mid_coords)
            index_distance_mid_tip = math.dist(index_mid_coords, index_tip_coords)
            index_distance_base_tip = math.dist(index_base_coords, index_tip_coords)

            # İki vektör arasındaki açıyı hesapla (Cosinus teoremi kullanılarak)
            index_angle = math.degrees(math.acos((index_distance_base_mid ** 2 + index_distance_mid_tip ** 2 - index_distance_base_tip ** 2) /
                                           (2 * index_distance_base_mid * index_distance_mid_tip)))

            # Açıyı ekrana yazdır
            cv2.putText(img, f"Index_Angle: {int(index_angle)}", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Eklemleri çiz
            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

            # Orta parmak 10. 11. ve 12. landmarklar arası açıları hesaplar
            # Orta parmağın eklemlerinin indexleri
            mid_base = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP]
            mid_mid = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_DIP]
            mid_tip = hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_TIP]

            # Orta parmağın eklemlerinin koordinatlarını alın
            mid_base_coords = (mid_base.x, mid_base.y)
            mid_mid_coords = (mid_mid.x, mid_mid.y)
            mid_tip_coords = (mid_tip.x, mid_tip.y)

            # İki eklemin koordinatları arasındaki mesafeyi ve açıyı hesapla
            mid_distance_base_mid = math.dist(mid_base_coords, mid_mid_coords)
            mid_distance_mid_tip = math.dist(mid_mid_coords, mid_tip_coords)
            mid_distance_base_tip = math.dist(mid_base_coords, mid_tip_coords)

            # İki vektör arasındaki açıyı hesapla (Cosinus teoremi kullanılarak)
            mid_angle = math.degrees(math.acos((mid_distance_base_mid ** 2 + mid_distance_mid_tip ** 2 - mid_distance_base_tip ** 2 ) /
                                     (2 * mid_distance_base_mid * mid_distance_mid_tip)))

            # Açıyı ekrana yazdır
            cv2.putText(img, f"Mid_Angle: {int(mid_angle)}", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Eklemleri çiz
            # mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)


    if hlms.multi_hand_landmarks:
        for handlandmarks in hlms.multi_hand_landmarks:

            for fingerNum, landmark in enumerate(handlandmarks.landmark):
                positionX, positionY = int(landmark.x * width), int(landmark.y * height)


            mpDraw.draw_landmarks(img, handlandmarks, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Camera", img)


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break