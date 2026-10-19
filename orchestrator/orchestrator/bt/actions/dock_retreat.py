"""Back straight away from a dock after manipulation, before Navigation."""
import math
from orchestrator.bt.actions.base_action import BaseAction
from orchestrator.bt.bt_core import NodeStatus

class DockRetreat(BaseAction):
    def __init__(self,node,distance:float=.20,speed:float=.05,timeout:float=20.,
                 scan_topic:str='/scan',odom_topic:str='/odom',cmd_vel_topic:str='/cmd_vel',
                 base_frame:str='base_link',footprint_profile:str='circle'):
        super().__init__(node,name='DockRetreat')
        self.distance,self.speed,self.timeout=map(float,(distance,speed,timeout))
        for value,low,high in ((self.distance,.05,.5),(self.speed,.02,.10),(self.timeout,5.,60.)):
            if not math.isfinite(value) or not low<=value<=high:raise ValueError('Invalid DockRetreat distance/speed/timeout')
        if footprint_profile not in ('circle','sg2_autonomy_base'):raise ValueError('Unknown footprint_profile')
        self.args=(self.distance,self.speed,self.timeout,scan_topic,odom_topic,cmd_vel_topic,base_frame,footprint_profile)
        self.runtime=None;self.failure=False
    def tick(self):
        if self.failure:return NodeStatus.FAILURE
        if self.runtime is None:
            try:
                from orchestrator.bt.docking_retreat import DockRetreatRuntime
                self.runtime=DockRetreatRuntime(self.node,*self.args);self.runtime.start()
            except Exception as exc:
                if self.runtime:self.runtime.close();self.runtime=None
                self.log_error(str(exc));self.failure=True;return NodeStatus.FAILURE
        if self.runtime.result is None:return NodeStatus.RUNNING
        return NodeStatus.SUCCESS if self.runtime.result else NodeStatus.FAILURE
    def pause(self):
        if self.runtime:self.runtime.pause()
    def resume(self):
        if self.runtime:self.runtime.resume()
    def reset(self):
        if self.runtime:self.runtime.close();self.runtime=None
        self.failure=False;super().reset()
    def destroy(self):self.reset()
