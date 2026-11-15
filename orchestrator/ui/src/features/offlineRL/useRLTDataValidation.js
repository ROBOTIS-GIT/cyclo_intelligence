import { useEffect, useRef, useState } from 'react';
import { validateRLTStage2Data, getRLTStage2DataValidation } from '../../utils/offlineRlApi';

export default function useRLTDataValidation(request, enabled) {
  const key = JSON.stringify(request);
  const activeKey = useRef(key);
  activeKey.current = key;
  const mounted = useRef(true);
  const pending = useRef(false);
  const [state, setState] = useState(null);
  const [attempted, setAttempted] = useState(false);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  const result = state?.key === key ? state.result : null;
  useEffect(() => {
    if (result?.status !== 'running') return undefined;
    let cancelled = false;
    let timer;
    const poll = async () => {
      try {
        const next = await getRLTStage2DataValidation(result.validation_id);
        if (cancelled || activeKey.current !== key) return;
        setState({ key, result: next });
      } catch (error) {
        if (!cancelled && activeKey.current === key) {
          setState({ key, result: { status: 'blocked', message: error.message } });
        }
      }
    };
    timer = setTimeout(poll, 1500);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [result, key]);
  const run = async () => {
    if (!enabled || pending.current || result?.status === 'running') return;
    pending.current = true;
    setAttempted(true);
    setState({ key, result: { status: 'starting', message: '데이터 검증을 시작합니다.' } });
    try {
      const next = await validateRLTStage2Data(request);
      if (mounted.current && activeKey.current === key) setState({ key, result: next });
    } catch (error) {
      if (mounted.current && activeKey.current === key) {
        setState({ key, result: { status: 'blocked', message: error.message } });
      }
    } finally { pending.current = false; }
  };
  return {
    result, run, attempted,
    ready: result?.status === 'ready',
    busy: ['starting', 'running'].includes(result?.status),
  };
}
