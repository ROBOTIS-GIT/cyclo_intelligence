from types import SimpleNamespace as S
import json
import numpy as np
import pytest
from orchestrator.bt.docking_vision import MarkerDetector
from orchestrator.bt.docking_diagnostics import DockDiagnostics
from orchestrator.bt.docking_core import DockConfig


def test_detection_failure_categories():
    detector=MarkerDetector(11,.18)
    detector.detector=S(detectMarkers=lambda image: ([],None,[]))
    image=np.zeros((240,320),np.uint8);k=[300,0,160,0,300,120,0,0,1]
    assert detector.detect(image,k,[0]*5) is None
    assert detector.diagnostics['reason']=='no_markers_detected'
    corner=np.array([[[100,100],[110,100],[110,110],[100,110]]],dtype=float)
    detector.detector=S(detectMarkers=lambda image: ([corner],np.array([[10]]),[]))
    assert detector.detect(image,k,[0]*5) is None
    assert detector.diagnostics['reason']=='target_id_absent'
    assert detector.diagnostics['detected_ids']==[10]
    detector.detector=S(detectMarkers=lambda image: ([corner],np.array([[11]]),[]))
    assert detector.detect(image,k,[0]*5) is None
    assert detector.diagnostics['reason']=='target_too_small'
    assert detector.diagnostics['min_edge_px']==10


def test_history_is_bounded_and_artifacts_keep_frame_metadata(tmp_path):
    diagnostics=DockDiagnostics(11,DockConfig(),'raw',tmp_path)
    image=np.zeros((24,32),np.uint8)
    for i in range(180):
        diagnostics.last_frame=0
        diagnostics.record({'image_stamp_s':i,'detection':{'reason':'target_too_small','min_edge_px':10}},image,image)
    assert len(diagnostics.events)==160
    assert len(diagnostics.frames)==32
    folder=tmp_path/'failure'
    diagnostics.write(folder,'Marker 11 too small',list(diagnostics.events),list(diagnostics.frames))
    report=json.loads((folder/'report.json').read_text())
    assert report['frames'][-1]['image_stamp_s']==179
    assert report['failure']=='Marker 11 too small'
    assert (folder/report['frames'][-1]['rectified_image']).exists()
    assert (folder/report['frames'][-1]['raw_image']).exists()
    assert (folder/report['frames'][-1]['overlay_image']).exists()
