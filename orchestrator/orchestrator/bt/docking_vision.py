"""OpenCV 4.6+ marker detection; no ROS dependency or world-position shortcut."""
import cv2
import numpy as np


class MarkerDetector:
    def __init__(self, marker_id, marker_size):
        if int(marker_id) != marker_id or not 0 <= marker_id < 50:
            raise ValueError('marker_id must be an integer in DICT_4X4_50 (0..49)')
        if not np.isfinite(marker_size) or not .05 <= marker_size <= 1.:
            raise ValueError('marker_size must be in [0.05, 1.0] metres')
        self.marker_id = int(marker_id)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = (cv2.aruco.DetectorParameters() if hasattr(cv2.aruco, 'ArucoDetector')
                       else cv2.aruco.DetectorParameters_create())
        self.params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.params) if hasattr(cv2.aruco, 'ArucoDetector') else None
        h = marker_size / 2
        self.object_points = np.array([[-h,h,0], [h,h,0], [h,-h,0], [-h,-h,0]], dtype=np.float64)

    def detect(self, image, matrix, distortion):
        corners, ids, _ = (self.detector.detectMarkers(image) if self.detector else
                           cv2.aruco.detectMarkers(image, self.dictionary, parameters=self.params))
        if ids is None:
            return None
        indices = np.flatnonzero(ids.reshape(-1) == self.marker_id)
        if len(indices) != 1:
            return None
        pixels = corners[int(indices[0])].reshape(4, 2).astype(np.float64)
        if min(np.linalg.norm(pixels[i]-pixels[(i+1)%4]) for i in range(4)) < 18:
            return None
        matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        distortion = np.asarray(distortion, dtype=np.float64)
        if not np.isfinite(matrix).all() or matrix[0,0] <= 0 or matrix[1,1] <= 0:
            raise ValueError('CameraInfo is not calibrated')
        ok, rvec, tvec = cv2.solvePnP(self.object_points, pixels, matrix, distortion,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or tvec[2,0] <= 0 or not np.isfinite(tvec).all():
            return None
        projected, _ = cv2.projectPoints(self.object_points, rvec, tvec, matrix, distortion)
        rms = float(np.sqrt(np.mean(np.sum((projected.reshape(4,2)-pixels)**2, axis=1))))
        if rms > 2.5:
            return None
        rotation, _ = cv2.Rodrigues(rvec)
        return tvec.reshape(3), rotation[:,2], pixels, rms


def rotation_matrix(q):
    values = np.array([q.x, q.y, q.z, q.w], dtype=float)
    if not np.isfinite(values).all() or np.linalg.norm(values) < 1e-8:
        raise ValueError('Invalid TF quaternion')
    x,y,z,w = values / np.linalg.norm(values)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def transform_pose(transform, position, normal):
    r = rotation_matrix(transform.rotation)
    p = transform.translation
    return r @ position + np.array([p.x,p.y,p.z]), r @ normal
