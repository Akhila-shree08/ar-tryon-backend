import cv2

class SurfaceModule:
    """
    NOTE:
    Real plane/surface tracking should be done in Unity using AR Foundation (ARCore).
    This module only provides basic frame info / debugging helpers.
    """

    def get_surface_info(self, bgr_frame):
        h, w = bgr_frame.shape[:2]

        # Optional: quick edge map statistic (debug)
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        edge_pixels = int((edges > 0).sum())

        return {
            "found": True,
            "mode": "surface_debug",
            "width": int(w),
            "height": int(h),
            "edge_pixels": edge_pixels,
            "note": "For real surface/plane AR use Unity AR Foundation (ARCore) back camera plane detection + raycast."
        }
