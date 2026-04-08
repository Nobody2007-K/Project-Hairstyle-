import cv2
import mediapipe as mp
import numpy as np
from config import IMG_SIZE, TOP_MARGIN, BOTTOM_MARGIN, SIDE_MARGIN

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def clamp(self, v, lo, hi):
        return max(lo, min(hi, v))
    
    def rotate(self, img, angle, center):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        out = cv2.warpAffine(img, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        return out, M
    
    def apply_transform(self, M, x, y):
        return (
            M[0,0] * x + M[0,1] * y + M[0,2],
            M[1,0] * x + M[1,1] * y + M[1,2]
        )
    
    def detect_and_crop(self, img_bgr):
        """
        Detect face and crop it for model input
        Returns: cropped face image or None
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        h, w = img_bgr.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        pts = [(p.x * w, p.y * h) for p in landmarks]
        
        # Align by outer eye corners
        lx, ly = pts[33]
        rx, ry = pts[263]
        angle = np.degrees(np.arctan2(ry - ly, rx - lx))
        
        rot, M = self.rotate(img_bgr, -angle, (w/2, h/2))
        rpts = [self.apply_transform(M, x, y) for x, y in pts]
        
        xs, ys = zip(*rpts)
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)
        
        fw, fh = x2 - x1, y2 - y1
        if fw < 5 or fh < 5:
            return None
        
        nx1 = int(self.clamp(x1 - SIDE_MARGIN * fw, 0, w))
        nx2 = int(self.clamp(x2 + SIDE_MARGIN * fw, 0, w))
        ny1 = int(self.clamp(y1 - TOP_MARGIN * fh, 0, h))
        ny2 = int(self.clamp(y2 + BOTTOM_MARGIN * fh, 0, h))
        
        crop = rot[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            return None
        
        crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        return crop
    
    def close(self):
        self.face_mesh.close()
