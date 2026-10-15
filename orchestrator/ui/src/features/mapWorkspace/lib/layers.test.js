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
