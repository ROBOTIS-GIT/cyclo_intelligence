"""TT-RTC status and build must resolve the same engine as runtime LOAD."""
import asyncio
from types import SimpleNamespace
import pytest
from supervisor_api.test_app import app


@pytest.mark.parametrize('mode,filename', [('async','dit_model_bf16.trt'),('sync','dit_model_bf16.trt'),('tt_rtc','dit_model_bf16_ttrtc.trt')])
def test_mode_selects_matching_engine(mode,filename):
    model='/workspace/model/groot/mock'
    assert app._resolve_groot_trt_paths(model,action_request_mode=mode) == (model,model+'/'+filename)


def test_tt_rtc_cannot_overwrite_ordinary_engine():
    with pytest.raises(app.HTTPException, match='') as error:
        app._resolve_groot_trt_paths('/workspace/model/groot/mock','dit_model_bf16.trt','tt_rtc')
    assert error.value.status_code == 400


def test_request_and_status_preserve_tt_rtc_mode(monkeypatch):
    request=app.TrtBuildRequest(model_path='/workspace/model/groot/mock',robot_type='ffw_sg2_rev1',action_request_mode='tt_rtc')
    assert request.action_request_mode == 'tt_rtc'
    monkeypatch.setattr(app,'_trt_status',lambda model,engine:(model,engine))
    result=asyncio.run(app.groot_trt_status(request.model_path,action_request_mode='tt_rtc'))
    assert result[1].endswith('/dit_model_bf16_ttrtc.trt')


def test_build_command_passes_mode_to_converter(monkeypatch,tmp_path):
    threads=[]
    class Thread:
        def __init__(self,**kwargs): threads.append(kwargs)
        def start(self): pass
    monkeypatch.setattr(app.threading,'Thread',Thread)
    monkeypatch.setattr(app,'_TRT_BUILD_JOBS',{})
    app._start_trt_build_job(str(tmp_path),str(tmp_path/'dit_model_bf16_ttrtc.trt'),'ffw_sg2_rev1','pick item',4096,False,action_request_mode='tt_rtc')
    command=threads[0]['args'][1]
    assert command[command.index('--action-request-mode')+1] == 'tt_rtc'
