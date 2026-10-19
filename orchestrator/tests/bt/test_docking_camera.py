from types import SimpleNamespace
import cv2
import numpy as np
import pytest
from orchestrator.bt.docking_camera import camera_source, detection_image, normalize_profile
from orchestrator.bt.docking_vision import MarkerDetector


def test_source_selection_and_explicit_override():
    node=SimpleNamespace(count_publishers=lambda _: 1)
    assert camera_source(node,'','','raw')[0]=='/zed/left/high_resolution/image_raw'
    node.count_publishers=lambda _:0
    assert camera_source(node,'','','raw')==('/zed/zed_node/left/image_rect_color','/zed/zed_node/left/camera_info','rectified')
    assert camera_source(node,'/custom/image','/custom/info','raw')==('/custom/image','/custom/info','raw')


def test_resize_calibration_and_no_fake_hd():
    image=np.zeros((720,1280),np.uint8)
    k=np.array([[1000.,0,640],[0,1000,360],[0,0,1.]])
    resized,scaled=detection_image(image,k,'VGA')
    assert resized.shape==(376,672)
    assert scaled[0,0]==525 and scaled[0,2]==336
    assert scaled[1,2]==188
    assert k[0,0]==1000
    with pytest.raises(ValueError,match='upscaling is disabled'):
        detection_image(resized,scaled,'HD720')
    assert detection_image(image,k,'HD720')[0] is image
    assert normalize_profile('standard')=='VGA'
    assert normalize_profile('high_resolution')=='HD720'


@pytest.mark.parametrize('profile',['VGA','HD720'])
def test_metric_marker_pose_preserved(profile):
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker=(cv2.aruco.generateImageMarker(dictionary,11,120) if hasattr(cv2.aruco,'generateImageMarker') else cv2.aruco.drawMarker(dictionary,11,120))
    image=np.full((720,1280),255,np.uint8);image[300:420,580:700]=marker
    k=np.array([[1000.,0,640],[0,1000,360],[0,0,1.]])
    image,k=detection_image(image,k,profile)
    found=MarkerDetector(11,.12).detect(image,k,np.zeros(5))
    assert found is not None
    assert found[0][2]==pytest.approx(1.,abs=.025)
