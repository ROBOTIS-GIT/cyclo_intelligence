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

import { useRef } from "react";

// Nav2 retains /plan after cancellation. A new leg must receive a fresh path.
export default function useActiveNavigationPlan(active, plan, legKey = "navigation") {
  const cache = useRef({ key: null, stale: plan });
  const key = active ? legKey : null;
  if (!active || cache.current.key !== key) cache.current = { key, stale: plan };
  return active && plan !== cache.current.stale ? plan : null;
}
