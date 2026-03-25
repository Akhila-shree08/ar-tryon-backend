import mediapipe as mp
import math


class PoseModule:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            smooth_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def get_landmarks(self, frame):
        results = self.pose.process(frame)

        if not results.pose_landmarks:
            return {"found": False}

        lm = results.pose_landmarks.landmark

        ls = lm[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        rs = lm[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        lh = lm[self.mp_pose.PoseLandmark.LEFT_HIP]
        rh = lm[self.mp_pose.PoseLandmark.RIGHT_HIP]

        # Body angle from shoulders (good for dress rotation)
        body_angle = math.degrees(math.atan2(rs.y - ls.y, rs.x - ls.x))

        conf = float(min(ls.visibility, rs.visibility))

        return {
            "found": True,
            "left_shoulder": [ls.x, ls.y, ls.z],
            "right_shoulder": [rs.x, rs.y, rs.z],
            "left_hip": [lh.x, lh.y, lh.z],
            "right_hip": [rh.x, rh.y, rh.z],
            "body_angle": float(body_angle),
            "conf": conf,
        }
