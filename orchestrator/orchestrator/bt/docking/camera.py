"""Camera source selection and calibrated detection resolution (no upscaling)."""
import cv2

RESOLUTIONS = {'VGA': (672, 376), 'HD720': (1280, 720)}


def normalize_profile(value):
    # Read older saved trees; the editor exposes only the two supported modes.
    key = str(value).strip().upper()
    key = {'STANDARD': 'VGA', 'HIGH_RESOLUTION': 'HD720'}.get(key, key)
    if key not in RESOLUTIONS:
        raise ValueError('camera_profile must be VGA or HD720')
    return key


def camera_source(node, image_topic, info_topic, mode):
    if image_topic and info_topic:
        return image_topic, info_topic, mode
    native = '/zed/left/high_resolution/'
    if node.count_publishers(native + 'image_raw'):
        return native + 'image_raw', native + 'camera_info', 'raw'
    return ('/zed/zed_node/left/image_rect_color',
            '/zed/zed_node/left/camera_info', 'rectified')


def detection_image(image, intrinsic, profile):
    width, height = RESOLUTIONS[normalize_profile(profile)]
    source_h, source_w = image.shape[:2]
    if source_w < width or source_h < height:
        raise ValueError(f'{profile} requires at least {width}x{height}; received {source_w}x{source_h}. '
                         'Configure the camera grab resolution and matching CameraInfo; upscaling is disabled.')
    if (source_w, source_h) == (width, height):
        return image, intrinsic
    # Scale image and calibrated pixel coordinates together, including the
    # slight aspect-ratio difference between ZED VGA and HD720.
    k = intrinsic.copy()
    k[0, :] *= width / source_w
    k[1, :] *= height / source_h
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), k
