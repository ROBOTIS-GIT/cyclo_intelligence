from types import SimpleNamespace as S
import cv2
import numpy as np
import pytest
from orchestrator.bt.docking_rectification import DockRectifier


def calibration():
    k=[300.,0,160,0,300,120,0,0,1]
    return S(width=320,height=240,k=k,d=[.12,-.04,0,0,0],r=np.eye(3).ravel(),
             p=[280.,0,158,0,0,280,118,0,0,0,1,0],distortion_model='plumb_bob',
             binning_x=0,binning_y=0,roi=S(x_offset=0,y_offset=0,width=0,height=0))


def test_raw_rectification_and_cached_maps_follow_calibration():
    info=calibration(); rect=DockRectifier(); image=np.arange(320*240,dtype=np.uint8).reshape(240,320)
    result,k,inverse=rect.prepare(image,info)
    maps=rect.maps
    expected=cv2.initUndistortRectifyMap(np.array(info.k).reshape(3,3),np.array(info.d),np.eye(3),k,(320,240),cv2.CV_32FC1)
    assert np.array_equal(result,cv2.remap(image,*expected,cv2.INTER_LINEAR))
    assert k[0,0]==280 and not np.array_equal(result,image)
    rect.prepare(image,info); assert rect.maps is maps
    info.d[0]=.2;rect.prepare(image,info);assert rect.maps is not maps


def test_rectified_input_uses_p_without_double_correcting():
    info=calibration();image=np.zeros((240,320),np.uint8)
    rect=DockRectifier('rectified');result,k,inverse=rect.prepare(image,info)
    assert result is image and rect.maps is None
    assert k[0,0]==280 # Raw K is 300 and D is deliberately nonzero.


def test_rectified_pose_is_rotated_back_to_original_camera():
    info=calibration();r,_=cv2.Rodrigues(np.array([0.,.1,0.]));info.r=r.ravel()
    _,k,inverse=DockRectifier().prepare(np.zeros((240,320),np.uint8),info)
    original=np.array([.2,.1,1.]);assert inverse @ (r @ original)==pytest.approx(original)


@pytest.mark.parametrize('field,value',[('p',[0]*12),('r',[0]*9),('binning_x',2),('d',[float('nan')]*5)])
def test_invalid_calibration_rejected(field,value):
    info=calibration();setattr(info,field,value)
    with pytest.raises(ValueError):DockRectifier().prepare(np.zeros((240,320),np.uint8),info)


def test_zero_distortion_gazebo_identity():
    info=calibration(); info.d=[0]*5;info.p=[300.,0,160,0,0,300,120,0,0,0,1,0]
    image=np.arange(320*240,dtype=np.uint8).reshape(240,320)
    result,_,_=DockRectifier().prepare(image,info)
    assert np.array_equal(image,result)
