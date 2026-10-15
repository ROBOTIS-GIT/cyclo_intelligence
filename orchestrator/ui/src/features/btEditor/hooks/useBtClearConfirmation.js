// Copyright 2026 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Author: Seongwoo Kim

import { useCallback, useEffect, useRef, useState } from 'react';

export default function useBtClearConfirmation() {
  const [clearTreeArmed, setClearTreeArmed] = useState(false);
  const clearTreeTimerRef = useRef(null);
  const clearTreeTargetRef = useRef(null);
  const disarmClearTree = useCallback(() => {
    if (clearTreeTimerRef.current) {
      clearTimeout(clearTreeTimerRef.current);
      clearTreeTimerRef.current = null;
    }
    clearTreeTargetRef.current = null;
    setClearTreeArmed(false);
  }, []);

  const armClearTree = useCallback((snapshot) => {
    if (clearTreeTimerRef.current) clearTimeout(clearTreeTimerRef.current);
    clearTreeTargetRef.current = snapshot;
    setClearTreeArmed(true);
    clearTreeTimerRef.current = setTimeout(() => {
      clearTreeTimerRef.current = null;
      clearTreeTargetRef.current = null;
      setClearTreeArmed(false);
    }, 4000);
  }, []);

  useEffect(() => () => {
    if (clearTreeTimerRef.current) clearTimeout(clearTreeTimerRef.current);
  }, []);

  return { clearTreeArmed, clearTreeTimerRef, clearTreeTargetRef,
    armClearTree, disarmClearTree };
}
