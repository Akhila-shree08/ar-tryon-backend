import mediapipe as mp


class FaceModule:
    def __init__(self):
        self.mp_face = mp.solutions.face_mesh

        # Better settings for real-time try-on
        self.face = self.mp_face.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # Landmarks (MediaPipe FaceMesh indices)
        # Eye outer corners are good for scale + rotation
        self.LEFT_EYE_OUTER = 33
        self.RIGHT_EYE_OUTER = 263

        # Nose bridge is usually more stable than nose tip for glasses anchor
        self.NOSE_BRIDGE = 168

        # Optional ear-ish points (can help for temple alignment)
        self.LEFT_EAR = 234
        self.RIGHT_EAR = 454

    def get_landmarks(self, frame):
        results = self.face.process(frame)

        if not results.multi_face_landmarks:
            return {"found": False}

        lm = results.multi_face_landmarks[0].landmark

        le = lm[self.LEFT_EYE_OUTER]
        re = lm[self.RIGHT_EYE_OUTER]
        nb = lm[self.NOSE_BRIDGE]
        left_ear = lm[self.LEFT_EAR]
        right_ear = lm[self.RIGHT_EAR]

        # FaceMesh doesn't give a simple "visibility" like pose, so we return a simple ok flag only.
        return {
            "found": True,
            "left_eye": [le.x, le.y, le.z],
            "right_eye": [re.x, re.y, re.z],
            "nose_bridge": [nb.x, nb.y, nb.z],
            "left_ear": [left_ear.x, left_ear.y, left_ear.z],
            "right_ear": [right_ear.x, right_ear.y, right_ear.z],
        }
