import { act, fireEvent, render, screen } from '@testing-library/react';
import InferenceDaggerPanel from './InferenceDaggerPanel';
let mockStatus;let mockBackend;const mockCommand=jest.fn();
jest.mock('../hooks/usePolicyBackendStatus',()=>({__esModule:true,default:()=>mockBackend}));
jest.mock('react-redux',()=>({useSelector:fn=>fn({ros:{rosbridgeUrl:'ws://test'},tasks:{robotType:'ffw_sg2_rev1'}})}));
jest.mock('../utils/rosConnectionManager',()=>({url:'ws://test',getCurrentConnection:()=>({isConnected:true})}));
jest.mock('../utils/daggerControl',()=>({DAGGER_INITIAL:{state:'IDLE',busy:false,profiles:[]},DaggerClient:function(){
  this.snapshot=()=>mockStatus;this.command=mockCommand;this.heartbeat=jest.fn();this.dispose=jest.fn();
}}));
beforeEach(()=>{mockCommand.mockClear();mockStatus={state:'IDLE',online:true,busy:false,profiles:[],gateway:{},robot_input_ready:true};
  mockBackend={readiness:{ready:true,state:'ready'},refreshStatus:jest.fn()};});

test('default preparation is observation-only and explicit checkbox enables motion',()=>{
  render(<InferenceDaggerPanel />);fireEvent.click(screen.getByRole('button',{name:'모델 준비'}));
  expect(mockCommand).toHaveBeenCalledWith('prepare',{profile:'784_786_789_790_final',motion:false});
});

test('release and phase confirmations are explicit while held',()=>{
  mockStatus={...mockStatus,state:'REQUEST_HOLD',busy:true,motion:true,phase:1};
  render(<InferenceDaggerPanel />);
  fireEvent.click(screen.getByRole('button',{name:'정렬·삽입 상태 확인 · 해제 허용'}));
  expect(mockCommand).toHaveBeenCalledWith('confirm_release',undefined);
});

test('active DAgger ownership locks other controls even if the tab is inactive',()=>{
  mockStatus={...mockStatus,state:'HUMAN',busy:true,motion:true,phase:0};const busy=jest.fn();
  render(<InferenceDaggerPanel isActive={false} onBusyChange={busy}/>);
  expect(busy).toHaveBeenCalledWith(true);expect(screen.getByRole('button',{name:'모델 준비'})).toBeDisabled();
});

test('disconnected sessions cannot issue resume or takeover',()=>{
  mockStatus={...mockStatus,state:'REQUEST_HOLD',busy:true,motion:true,online:false};
  render(<InferenceDaggerPanel />);expect(screen.getByRole('button',{name:'로봇에게 반환'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'리더 암으로 개입'})).toBeDisabled();
});

test('stopped backend disables preparation and offers server start',()=>{
  mockBackend.readiness={ready:false,state:'stopped',message:'Policy Docker is off'};
  render(<InferenceDaggerPanel/>);
  expect(screen.getByRole('button',{name:'모델 준비'})).toBeDisabled();
  expect(screen.getByRole('button',{name:'GR00T 서버 시작'})).toBeEnabled();
});

test('missing local weights disable prepare and show the checkpoint error',()=>{
  mockStatus.profiles=[{key:'784_786_789_790_final',fps:30,available:false,error:'Missing shard',model_path:'/workspace/model/groot/local'}];
  render(<InferenceDaggerPanel/>);
  expect(screen.getByRole('button',{name:'모델 준비'})).toBeDisabled();
  expect(screen.getByText('Missing shard')).toBeInTheDocument();
});

test('motion requires a fresh ready exclusive gateway',()=>{
  render(<InferenceDaggerPanel/>);
  expect(screen.getByRole('checkbox',{name:'실제 로봇을 움직이며 수집'})).toBeDisabled();
});

test('a legacy direct gateway cannot enable robot collection',()=>{
  mockStatus={...mockStatus,gateway_fresh:true,gateway:{ready:true,route_mode:'legacy_direct'}};
  render(<InferenceDaggerPanel/>);
  expect(screen.getByRole('checkbox',{name:'실제 로봇을 움직이며 수집'})).toBeDisabled();
});

test('missing robot inputs disable preparation before loading',()=>{
  mockStatus.robot_input_ready=false;
  render(<InferenceDaggerPanel/>);
  expect(screen.getByRole('button',{name:'모델 준비'})).toBeDisabled();
  expect(screen.getByText(/로봇 관절 입력이 없습니다/)).toBeInTheDocument();
});

test('standard mux routing enables explicit robot mode preparation',()=>{
  mockStatus={...mockStatus,gateway_fresh:true,gateway:{ready:true,route_mode:'ros2_topic_tools'}};
  render(<InferenceDaggerPanel/>);
  fireEvent.click(screen.getByRole('checkbox',{name:'실제 로봇을 움직이며 수집'}));
  fireEvent.click(screen.getByRole('button',{name:'모델 준비'}));
  expect(mockCommand).toHaveBeenCalledWith('prepare',{profile:'784_786_789_790_final',motion:true});
});

test('expert review approval does not carry over to the next session',()=>{
  jest.useFakeTimers();let view;
  try {
    mockStatus={...mockStatus,state:'HUMAN',busy:true,session_id:'first'};
    view=render(<InferenceDaggerPanel/>);
    const review=screen.getByRole('checkbox',{name:'사람 교정 구간을 검수했으며 학습에 사용'});
    fireEvent.click(review);fireEvent.click(screen.getByRole('button',{name:'성공 저장'}));
    expect(mockCommand).toHaveBeenLastCalledWith('finish',{outcome:'success',human_review:true});
    mockStatus={...mockStatus,session_id:'second'};
    act(()=>{jest.advanceTimersByTime(100);});
    expect(review).not.toBeChecked();
    fireEvent.click(screen.getByRole('button',{name:'성공 저장'}));
    expect(mockCommand).toHaveBeenLastCalledWith('finish',{outcome:'success',human_review:false});
  } finally { view?.unmount();jest.useRealTimers(); }
});
