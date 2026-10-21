import React, { useEffect, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import rosConnectionManager from '../utils/rosConnectionManager';
import { DaggerClient, DAGGER_INITIAL } from '../utils/daggerControl';
import usePolicyBackendStatus from '../hooks/usePolicyBackendStatus';

const LABELS = { IDLE:'대기', PREPARING:'모델 준비 중', READY:'수집 준비 완료', AUTO:'자율 실행',
  REQUEST_HOLD:'개입 요청 · 자세 유지', HUMAN:'리더 암 조작', RESUME_CHECK:'새 관측으로 반환 확인',
  SAVING:'저장 중', STOPPED:'수집 종료', ERROR:'준비 실패' };
const DEFAULT_PROFILE = '784_786_789_790_final';

export default function InferenceDaggerPanel({ isActive = true, externalBusy = false, onBusyChange }) {
  const url = useSelector((s) => s.ros.rosbridgeUrl);
  const robot = useSelector((s) => s.tasks.robotType);
  const client = useRef(null);
  const busyCallback = useRef(onBusyChange);
  busyCallback.current = onBusyChange;
  const activeRef = useRef(isActive);
  activeRef.current = isActive;
  const [status, setStatus] = useState(DAGGER_INITIAL);
  const [profile, setProfile] = useState(DEFAULT_PROFILE);
  const [motion, setMotion] = useState(false);
  const [review, setReview] = useState(false);
  const [error, setError] = useState('');
  const [waiting, setWaiting] = useState(null);
  const backend = usePolicyBackendStatus('groot', { enabled: isActive, intervalMs: 3000 });
  const [backendPending, setBackendPending] = useState(false);
  const backendPendingRef = useRef(false);
  backendPendingRef.current = backendPending;
  const waitingRef = useRef(null);
  const sendTime = useRef(0);

  useEffect(() => { setReview(false); }, [status.session_id]);

  useEffect(() => {
    const tick = () => {
      const ros = rosConnectionManager.url === url ? rosConnectionManager.getCurrentConnection() : null;
      if (ros && ros.isConnected && client.current?.ros !== ros) {
        client.current?.dispose();
        client.current = new DaggerClient(ros);
      }
      const next = client.current?.snapshot() || DAGGER_INITIAL;
      setStatus(next);
      busyCallback.current?.(next.busy || backendPendingRef.current);
      client.current?.heartbeat(activeRef.current && !document.hidden);
      if (waitingRef.current && next.request_id === waitingRef.current) {
        waitingRef.current = null; setWaiting(null);
      } else if (waitingRef.current && Date.now()-sendTime.current > 5000) {
        waitingRef.current = null; setWaiting(null);
        setError('요청 응답을 확인하지 못했습니다. 현재 상태를 확인한 뒤 다시 조작하세요.');
      }
    };
    tick();
    const timer = window.setInterval(tick, 100);
    return () => { window.clearInterval(timer); client.current?.dispose(); client.current = null; };
  }, [url]);

  const send = (op, fields) => {
    try {
      setError('');
      const id = client.current.command(op, fields);
      waitingRef.current = id; sendTime.current = Date.now(); setWaiting(id);
    } catch (e) { setError(e.message); }
  };
  const supported = /^ffw_sg2(?:_rev1)?$/i.test(robot);
  const blocked = !status.online || !!waiting || !!status.pending || backendPending || externalBusy || !supported;
  const selectedProfile = status.profiles.find((p) => p.key === profile);
  const gatewayReady = status.gateway_fresh && status.gateway?.ready && status.gateway?.route_mode === 'ros2_topic_tools';
  const startBackend = async () => {
    setBackendPending(true); setError('');
    try {
      const response = await fetch('/api/backends/groot/start', { method: 'POST' });
      const data = await response.json();
      if (!response.ok || data.ok === false) throw new Error(data.detail || data.message || 'GR00T 시작 실패');
      await backend.refreshStatus();
    } catch (e) { setError(e.message); }
    finally { setBackendPending(false); }
  };
  const held = ['REQUEST_HOLD','HUMAN'].includes(status.state);
  const finished = ['IDLE','STOPPED','ERROR'].includes(status.state);
  const button = 'rounded-lg border px-3 py-2 text-sm font-medium disabled:opacity-40 disabled:cursor-not-allowed';
  const phaseNames = status.profile?.startsWith('754_')
    ? ['정렬 · 홀더 위 배치','해제 · 안착','안전 후퇴'] : ['정렬 · 삽입','해제 · 안착','안전 후퇴'];

  return <div className="w-full p-4 space-y-4" onKeyDown={(e) => e.stopPropagation()} onKeyUp={(e) => e.stopPropagation()}>
    <div><h2 className="text-lg font-semibold">DAgger 교정 수집</h2>
      <p className="mt-1 text-sm text-gray-600">GR00T 실행 중 필요한 구간을 리더 암으로 교정하고, 교정 데이터와 개입 이유를 함께 저장합니다.</p></div>
    <div role="status" aria-live="polite" className="rounded-lg bg-blue-50 p-3 text-sm">
      <strong>{status.online ? LABELS[status.state] || status.state : 'DAgger 서비스 연결 대기'}</strong>
      {status.motion && <span className="ml-2 text-red-700">실제 로봇</span>}
      {status.instruction && <p className="mt-1">{status.instruction}</p>}
      {status.reason && <p className="mt-1 text-amber-800">요청 이유: {status.reason}</p>}
      <p className="mt-1">개입 {status.interventions || 0}회 · 사람 조작 {status.human_seconds || 0}초</p>
    </div>
    {(error || status.error) && <p role="alert" className="text-sm text-red-700">{error || status.error}</p>}
    {finished && status.model_owned && !status.pending && <button className={button}
      disabled={blocked} onClick={() => send('unload')}>모델 해제 재시도</button>}
    {!supported && <p className="text-sm text-amber-700">SG2 Rev1을 선택해주세요.</p>}
    <label className="block text-sm">학습 모델·작업 전략
      <select aria-label="DAgger profile" className="mt-1 w-full rounded border p-2" value={profile}
        disabled={status.busy || !!waiting} onChange={(e) => setProfile(e.target.value)}>
        {(status.profiles.length ? status.profiles : [
          {key:DEFAULT_PROFILE, fps:30}, {key:'784_786_789_790_best_validation',fps:30},
          {key:'754_760_761_762_773_final',fps:15}, {key:'754_760_761_762_773_best_validation',fps:15},
        ]).map((p) => <option key={p.key} value={p.key} disabled={p.available === false}>
          {p.key} · {p.fps} FPS{p.available === false ? ' · 파일 확인 필요' : ''}</option>)}
      </select>
    </label>
    {selectedProfile && <div className="rounded border p-3 text-xs space-y-1">
      <p>{selectedProfile.strategy === 'release_and_drop' ? '홀더 위 배치 → 해제·낙하 → 후퇴' : '삽입 → 해제·안착 → 후퇴'}
        {' · '}예측 24 / 초기 실행 {selectedProfile.execute_steps} action</p>
      <p className="break-all">로컬 체크포인트: {selectedProfile.model_path}</p>
      <p>{selectedProfile.available ? '로컬 파일·모달리티 계약 확인됨' : selectedProfile.error}</p>
    </div>}
    <div className="flex items-center gap-2 text-sm">
      <span>GR00T: {backend.readiness.ready ? '연결 준비 완료' : backend.readiness.message}</span>
      {backend.readiness.state === 'stopped' && <button className={button}
        disabled={blocked || status.busy || !!status.admission_reason} onClick={startBackend}>GR00T 서버 시작</button>}
    </div>
    <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={motion}
      disabled={status.busy || !!waiting || (!motion && !gatewayReady)} onChange={(e) => setMotion(e.target.checked)} />실제 로봇을 움직이며 수집</label>
    <p className="text-xs text-gray-600">체크하지 않으면 정책 출력만 기록합니다. 실제 제어 연결: {gatewayReady ? '준비 완료' : '중재기 상태 확인 필요'}.</p>
    {status.online && !status.robot_input_ready && <p className="text-sm text-amber-700">로봇 관절 입력이 없습니다. 로봇 연결·bringup을 먼저 확인해주세요.</p>}
    {finished && status.admission_reason && <p className="text-sm text-amber-700">{status.admission_reason}</p>}
    {status.sensor_reason && <p className="text-xs text-amber-700">입력 확인: {status.sensor_reason}</p>}
    <div className="grid grid-cols-2 gap-2">
      <button className={button} disabled={blocked || !finished || status.busy || !!status.admission_reason || !status.robot_input_ready || !backend.readiness.ready || selectedProfile?.available === false || (motion && !gatewayReady)}
        onClick={() => send('prepare',{profile,motion})}>모델 준비</button>
      <button className={`${button} bg-blue-600 text-white`} disabled={blocked || status.state !== 'READY' || !!status.sensor_reason}
        onClick={() => send('start')}>{status.motion ? '로봇 수집 시작' : '관찰 수집 시작'}</button>
      <button className={`${button} bg-amber-100`} disabled={!status.online || !['AUTO','HUMAN','RESUME_CHECK'].includes(status.state)}
        onClick={() => send('hold')}>개입 요청 · Hold</button>
      <button className={button} disabled={blocked || !status.motion || !['AUTO','REQUEST_HOLD'].includes(status.state)}
        onClick={() => send('takeover')}>리더 암으로 개입</button>
      <button className={button} disabled={blocked || !held}
        onClick={() => send('resume')}>로봇에게 반환</button>
      <button className={button} disabled={blocked || !held || status.phase >= 2}
        onClick={() => send('next_phase')}>{status.phase === 1 ? '안착 확인 → 후퇴' : '배치 확인 → 해제'}</button>
    </div>
    <ol aria-label="DAgger phases" className="flex gap-1 text-xs">
      {phaseNames.map((name,i) => <li key={name} className={`flex-1 rounded p-2 ${status.phase === i ? 'bg-blue-100 text-blue-800' : 'bg-gray-100'}`}>{i+1}. {name}</li>)}
    </ol>
    {status.phase === 1 && <button className={`${button} w-full`} disabled={blocked || !held || status.release_confirmed}
      onClick={() => send('confirm_release')}>{status.release_confirmed ? '해제 조건 확인됨' : '정렬·삽입 상태 확인 · 해제 허용'}</button>}
    <p className="text-xs text-gray-600">입력·관절 이상과 진행 정체는 자동으로 개입을 요청합니다. 해제 전 물체 상태와 단계 완료는 사람이 확인합니다. 학습된 개입 예측기: 데이터 수집 후 shadow 평가.</p>
    {held && status.reason && <div className="grid grid-cols-2 gap-2">
      <button className={button} disabled={blocked} onClick={() => send('gate_feedback',{needed:true})}>개입 요청 적절함</button>
      <button className={button} disabled={blocked} onClick={() => send('gate_feedback',{needed:false})}>개입 요청 불필요</button>
    </div>}
    {status.shadow_score && <p className="text-xs text-gray-600">개입 예측 shadow 점수: {status.shadow_score.need_help_score.toFixed(2)} · 실제 전환에는 사용하지 않음</p>}
    <label className="flex gap-2 text-sm"><input type="checkbox" checked={review} onChange={(e) => setReview(e.target.checked)} />사람 교정 구간을 검수했으며 학습에 사용</label>
    <div className="grid grid-cols-3 gap-2">
      {['success','failure','aborted'].map((outcome,i) => <button key={outcome} className={button}
        disabled={blocked || finished || status.state === 'SAVING'} onClick={() => send('finish',{outcome,human_review:review})}>
        {['성공 저장','실패 저장','중단 저장'][i]}</button>)}
    </div>
    {status.gateway?.latched && finished && <div className="space-y-2 text-sm">
      <p className="text-amber-700">로봇은 Hold를 유지합니다. 리더 팔·그리퍼를 현재 로봇에 맞춘 후 기존 제어로 반환하세요.</p>
      <button className={`${button} w-full`} disabled={blocked || !status.gateway.aligned}
        onClick={() => send('release')}>기존 제어로 반환</button></div>}
    {status.session_path && <p className="break-all text-xs text-gray-500">세션 기록: {status.session_path}</p>}
    <button className={`${button} w-full`} disabled={blocked || status.state !== 'STOPPED' || !status.session_path}
      onClick={() => send('export')}>교정 학습 데이터 내보내기</button>
    {status.export_path && <p className="break-all text-xs text-green-700">내보내기: {status.export_path}</p>}
  </div>;
}
