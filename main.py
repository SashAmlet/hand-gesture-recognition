import cv2
import mediapipe as mp
import numpy as np

def calculateDistance(point1, point2):
    """
    Calculate the Euclidean distance between two points.
    """
    return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

def recognize_O_feature0(hand_landmarks):
    """
    Check if the thumb tip and index tip are close enough to form an 'O' shape.
    """
    if hand_landmarks:
        thumb_tip, index_tip = hand_landmarks[0].landmark[4], hand_landmarks[0].landmark[8]
        return calculateDistance(thumb_tip, index_tip) < 0.03

def recognize_O_feature1(hand_landmarks):
    """
    Check if the middle, ring, and pinky fingers are open.
    """
    def fingerIsOpen(TIP, DIP):
        return TIP.y < DIP.y

    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark
        fingers = [(12, 11), (16, 15), (20, 19)]  # (TIP, DIP) pairs for middle, ring, and pinky fingers
        return all(fingerIsOpen(landmarks[tip], landmarks[dip]) for tip, dip in fingers)

def recognize_O_feature2(hand_landmarks):
    """
    Check if the hand orientation matches the required normal vector condition.
    """
    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark

        wrist = np.array([landmarks[0].x,
                        landmarks[0].y,
                        landmarks[0].z])

        index_mcp = np.array([landmarks[5].x,
                                landmarks[5].y,
                                landmarks[5].z])

        pinky_mcp = np.array([landmarks[17].x,
                            landmarks[17].y,
                            landmarks[17].z])

        # Calculate vectors from wrist to index MCP and pinky MCP
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist

        # Calculate the normal vector to the plane formed by the wrist, index MCP, and pinky MCP
        normal = np.cross(v1, v2)

        # Define the reference vector pointing along the negative Z-axis
        Oz = np.array([0, 0, -1])

        # Calculate the dot product of the normal vector and the reference vector
        normal_z = normal @ Oz

        # Print the normal_z value with 5 decimal places
        # print(f"{normal_z:.5f}")
        if normal_z > 0 and normal_z < 0.007:
            return True
        
def recognize_O_feature3(hand_landmarks):
    """
    Check if the distances between pinky-ring and ring-middle fingers are above a threshold.
    """
    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark

        pinky_ring_distance = calculateDistance(landmarks[20], landmarks[15])
        ring_middle_distance = calculateDistance(landmarks[16], landmarks[12])
        
        return pinky_ring_distance > 0.02 and ring_middle_distance > 0.02
        
def recognize_O_feature4(hand_landmarks):
    """
    Check if the distance between index MCP and index tip is above a threshold.
    """
    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark
        
        pinky_pinky_distance = calculateDistance(landmarks[8], landmarks[5])
      
        return pinky_pinky_distance > 0.035




def drawHandLandmarks(image, hand_landmarks):
    """
    Draw hand landmarks on the image.
    """
    if hand_landmarks:
        for landmarks in hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, landmarks, mp.solutions.hands.HAND_CONNECTIONS)

def main():
    """
    Main function to capture video, process hand landmarks, and recognize the 'O' gesture.
    """
    cap = cv2.VideoCapture(1)
    cap.set(3, 800)
    cap.set(4, 800)

    hands = mp.solutions.hands.Hands(max_num_hands=2)

    while True:
        _, image = cap.read()
        image = cv2.flip(image, 1)
        results = hands.process(image)
        hand_landmarks = results.multi_hand_landmarks

        drawHandLandmarks(image, hand_landmarks)

        if (recognize_O_feature0(hand_landmarks) and recognize_O_feature1(hand_landmarks) and 
            recognize_O_feature2(hand_landmarks) and recognize_O_feature3(hand_landmarks) and 
            recognize_O_feature4(hand_landmarks)):
            cv2.putText(image, "O", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "Not O", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Output", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()