import cv2
import mediapipe
import pyttsx3
import math

class HandGestureDetector:
    def __init__(self):
        self.camera = cv2.VideoCapture(0)
        self.engine = pyttsx3.init()
        self.mpHands = mediapipe.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mediapipe.solutions.drawing_utils
        self.finger_names = ["Thumb", "Index", "Middle", "Ring", "Little"]

    def process_hand_gestures(self):
        #el işaretlerini işlemek için ana işlevi gerçekleştirir.
        while True:
            success, img = self.camera.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hlms = self.hands.process(imgRGB)
            height, width, channel = img.shape

            if hlms.multi_hand_landmarks:
                for handlandmarks in hlms.multi_hand_landmarks:
                    parts = self.extract_landmark_parts(handlandmarks, width, height)
                    thumb_closed = self.is_thumb_closed(parts)
                    index_finger_closed = self.is_finger_closed(parts, 8, 6)
                    middle_finger_closed = self.is_finger_closed(parts, 12, 10)
                    ring_finger_closed = self.is_finger_closed(parts, 16, 14)
                    little_finger_closed = self.is_finger_closed(parts, 20, 18)
                    self.display_hand_gesture_status(img, thumb_closed, index_finger_closed, middle_finger_closed, ring_finger_closed, little_finger_closed)
                    self.draw_landmarks(img, handlandmarks)
                    self.display_finger_angles(img, parts)

            cv2.imshow("Camera", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.camera.release()
        cv2.destroyAllWindows()

    def extract_landmark_parts(self, handlandmarks, width, height):
        #el işareti landmark noktalarını (x, y) koordinatları ile birlikte çıkarır.

        parts = []
        for id, lm in enumerate(handlandmarks.landmark):
            cx, cy = int(lm.x * width), int(lm.y * height)
            parts.append((id, cx, cy))
        return parts



    def is_finger_closed(self, parts, tip_id, dip_id):
        #Parmakların kapalı olup olmadığını kontrol eder
        return parts[tip_id][2] < parts[dip_id][2]

    def is_thumb_closed(self, parts):
        #Başparmağın kapalı olup olmadığını kontrol eder
        return parts[4][2] < parts[3][2]

    def display_hand_gesture_status(self, img, thumb_closed, index_finger_closed, middle_finger_closed, ring_finger_closed, little_finger_closed):
        #Eldeki işaretlere göre hastalık durumunu belirler ve ekranda gösterir.
        if thumb_closed and index_finger_closed and middle_finger_closed and ring_finger_closed and little_finger_closed:
            message = "You're not sick"
        else:
            message = "You're sick"
        cv2.putText(img, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def draw_landmarks(self, img, handlandmarks):
        # El işareti landmark noktalarını ve bağlantılarını görüntü üzerine çizer.
        self.mpDraw.draw_landmarks(img, handlandmarks, self.mpHands.HAND_CONNECTIONS)

    def display_finger_angles(self, img, parts):
        #Parmakların açılarını hesaplar ve ekranda gösterir.
        angles = self.calculate_finger_angles(parts)
        for i, angle in enumerate(angles):
            if angle > 90:
                angle_text = "90"
            elif angle > 45:
                angle_text = "45"
            else:
                angle_text = "0"
            cv2.putText(img, f"{self.finger_names[i]} Angle: {angle_text}", (50, 80 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    def calculate_finger_angles(self, parts):
        #Parmakların açılarını hesaplar.
        angles = []
        for finger_id in range(5):
            if finger_id == 0:
                angle = self.calculate_angle(parts[0][1:], parts[1][1:], parts[2][1:])
            else:
                angle = self.calculate_angle(parts[finger_id * 4][1:], parts[finger_id * 4 + 1][1:], parts[finger_id * 4 + 2][1:])
            angles.append(angle)
        return angles

    def calculate_angle(self, point1, point2, point3):
        #Üç nokta arasındaki açıyı hesaplar.
        angle_radians = math.atan2(point3[1] - point2[1], point3[0] - point2[0]) - math.atan2(point1[1] - point2[1], point1[0] - point2[0])
        angle_degrees = math.degrees(angle_radians)
        return abs(angle_degrees)


if __name__ == "__main__":
    detector = HandGestureDetector()
    detector.process_hand_gestures()

