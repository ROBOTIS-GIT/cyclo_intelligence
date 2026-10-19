"""Calibrated left-camera rectification, cached per CameraInfo configuration."""
import cv2
import numpy as np


class DockRectifier:
    def __init__(self, mode='raw'):
        if mode not in ('raw', 'rectified'):
            raise ValueError('image_mode must be raw or rectified')
        self.mode, self.key, self.maps = mode, None, None

    def prepare(self, image, info):
        if image.shape[:2] != (info.height, info.width):
            raise ValueError('Image and CameraInfo dimensions do not match')
        if info.binning_x > 1 or info.binning_y > 1 or info.roi.x_offset or info.roi.y_offset:
            raise ValueError('Docking requires full-resolution CameraInfo without ROI offsets/binning')
        if (info.roi.width not in (0, info.width) or info.roi.height not in (0, info.height)):
            raise ValueError('Docking does not support cropped CameraInfo')
        k = np.asarray(info.k, dtype=float).reshape(3, 3)
        d = np.asarray(info.d, dtype=float)
        rotation = np.asarray(info.r, dtype=float).reshape(3, 3)
        projection = np.asarray(info.p, dtype=float).reshape(3, 4)
        intrinsic = projection[:, :3].copy()
        if not all(np.isfinite(v).all() for v in (k, d, rotation, projection)):
            raise ValueError('Non-finite camera calibration')
        if intrinsic[0,0] <= 0 or intrinsic[1,1] <= 0 or not np.allclose(intrinsic[2], [0,0,1]):
            raise ValueError('CameraInfo.P must contain calibrated rectified intrinsics')
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
            raise ValueError('CameraInfo.R must be a valid rectification rotation')
        if not np.allclose(projection[:,3], 0):
            raise ValueError('Docking requires the left/monocular camera (P translation must be zero)')
        if self.mode == 'rectified':
            # D describes the RAW camera even when the input topic is rectified.
            return image, intrinsic, rotation.T
        if k[0,0] <= 0 or k[1,1] <= 0 or not np.allclose(k[2], [0,0,1]):
            raise ValueError('CameraInfo.K must contain calibrated raw intrinsics')
        if info.distortion_model not in ('plumb_bob', 'rational_polynomial', 'equidistant', ''):
            raise ValueError('Unsupported raw camera distortion model')
        key = (image.shape, info.distortion_model, k.tobytes(), d.tobytes(), rotation.tobytes(), intrinsic.tobytes())
        if key != self.key:
            size = (info.width, info.height)
            if info.distortion_model == 'equidistant':
                if d.size != 4: raise ValueError('Equidistant distortion requires four coefficients')
                maps = cv2.fisheye.initUndistortRectifyMap(k, d, rotation, intrinsic, size, cv2.CV_32FC1)
            else:
                if d.size not in (0, 4, 5, 8, 12, 14):
                    raise ValueError('Invalid distortion coefficient count')
                maps = cv2.initUndistortRectifyMap(k, d, rotation, intrinsic, size, cv2.CV_32FC1)
            self.maps, self.key = maps, key
        rectified = cv2.remap(image, *self.maps, interpolation=cv2.INTER_LINEAR)
        # solvePnP returns the RECTIFIED optical coordinates; TF uses the
        # original camera optical frame. Undo R before applying that TF.
        return rectified, intrinsic, rotation.T
