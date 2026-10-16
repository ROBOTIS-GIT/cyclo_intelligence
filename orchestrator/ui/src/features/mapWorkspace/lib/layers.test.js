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

import { LAYER_PRESETS, STAGE_LAYER_IDS } from "./layers";
import { STAGE_COLLECT, STAGE_OPERATE, STAGE_RUN } from "./stages";
test.each([STAGE_COLLECT, STAGE_OPERATE])("%s enables just the four requested map layers",stage=>{
 expect(STAGE_LAYER_IDS[stage]).toEqual(["map","scan","robotModel","globalPlan"]);
 expect(Object.entries(LAYER_PRESETS[stage]).filter(([,v])=>v).map(([k])=>k).sort())
  .toEqual(["globalPlan","map","robotModel","scan"]);
});
test("Run retains its diagnostic layers",()=>{
 expect(STAGE_LAYER_IDS[STAGE_RUN]).toContain("globalCostmap");
 expect(STAGE_LAYER_IDS[STAGE_RUN]).toContain("tf");
});
