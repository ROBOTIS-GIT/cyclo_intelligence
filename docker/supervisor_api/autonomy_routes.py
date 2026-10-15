"""HTTP admission and plan endpoints; no execution loops."""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from .autonomy_config import ObservationPlan
from .navigation_spots import SpotPose


class AutonomyStart(BaseModel):
    map_name: str = Field(min_length=1, max_length=128)
    model_id: str = Field(min_length=1, max_length=200)
    localized: bool = False
    plan_revision: int = Field(ge=0)


class ObservationSpotWrite(BaseModel):
    label: str = Field(min_length=1, max_length=128)
    pose: SpotPose
    plan_revision: int = Field(ge=0)


class ConversationRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    request_id: str = Field(min_length=1, max_length=80)


class Instruction(BaseModel):
    text: str = Field(max_length=1000)


def create_router(controller):
    router = APIRouter(prefix='/autonomy', tags=['autonomy'])
    @router.get('/status')
    def status(): return controller.status()
    @router.post('/start')
    def start(request: AutonomyStart): return controller.start(request)
    @router.post('/stop')
    def stop(): return controller.stop()
    @router.post('/pause')
    def pause(): return controller.pause('Operator pause', source='manual')
    @router.post('/resume')
    def resume(): return controller.resume(source='manual')
    @router.post('/instruction')
    def instruction(request: Instruction): return controller.instruction(request.text)
    @router.get('/conversation')
    def conversation(): return controller.conversation.status()
    @router.post('/conversation')
    def converse(request: ConversationRequest): return controller.conversation.submit(request.text, request.request_id)
    @router.get('/maps/{map_name}')
    def configuration(map_name: str): return controller.repository.view(map_name)
    @router.put('/maps/{map_name}')
    def save(map_name: str, request: ObservationPlan):
        with controller.lock:
            if controller.owned: raise HTTPException(409, 'Stop autonomy before editing observation settings')
            try: controller.repository.save(map_name, request)
            except ValueError as exc: raise HTTPException(422, str(exc)) from exc
        return controller.repository.view(map_name)
    @router.post('/maps/{map_name}/spots')
    def create_spot(map_name: str, request: ObservationSpotWrite):
        with controller.lock:
            if controller.owned: raise HTTPException(409, 'Stop autonomy before editing observation spots')
            return controller.repository.write_observation_spot(map_name, request)
    @router.patch('/maps/{map_name}/spots/{spot_id}')
    def update_spot(map_name: str, spot_id: str, request: ObservationSpotWrite):
        with controller.lock:
            if controller.owned: raise HTTPException(409, 'Stop autonomy before editing observation spots')
            return controller.repository.write_observation_spot(map_name, request, spot_id)
    @router.delete('/maps/{map_name}/spots/{spot_id}')
    def delete_spot(map_name: str, spot_id: str, plan_revision: int = Query(ge=0)):
        with controller.lock:
            if controller.owned: raise HTTPException(409, 'Stop autonomy before deleting saved spots')
            controller.repository.delete_spot(map_name, spot_id, plan_revision)
            return controller.repository.view(map_name)
    return router
