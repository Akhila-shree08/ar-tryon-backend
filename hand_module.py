import mediapipe as mp
import math


class HandModule:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        self.WRIST = 0
        self.INDEX_MCP = 5
        self.MIDDLE_MCP = 9
        self.PINKY_MCP = 17

    def get_landmarks(self, frame):
        results = self.hands.process(frame)

        if not results.multi_hand_landmarks:
            return {"found": False}

        lm = results.multi_hand_landmarks[0].landmark

        wrist = lm[self.WRIST]
        index_mcp = lm[self.INDEX_MCP]
        middle_mcp = lm[self.MIDDLE_MCP]
        pinky_mcp = lm[self.PINKY_MCP]

        # Angle from wrist -> middle_mcp (more stable for watches)
        angle = math.degrees(math.atan2(middle_mcp.y - wrist.y, middle_mcp.x - wrist.x))

        # Confidence from handedness classifier if available
        conf = 1.0
        if results.multi_handedness and len(results.multi_handedness) > 0:
            conf = float(results.multi_handedness[0].classification[0].score)

        return {
            "found": True,
            "wrist": [wrist.x, wrist.y, wrist.z],
            "index_mcp": [index_mcp.x, index_mcp.y, index_mcp.z],
            "middle_mcp": [middle_mcp.x, middle_mcp.y, middle_mcp.z],
            "pinky_mcp": [pinky_mcp.x, pinky_mcp.y, pinky_mcp.z],
            "hand_angle": float(angle),
            "conf": conf,
        }
