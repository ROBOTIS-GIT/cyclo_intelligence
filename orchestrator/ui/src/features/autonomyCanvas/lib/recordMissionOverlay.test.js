import { recordMissionOverlay } from "./recordMissionOverlay";
const source={id:"observe",pose:{x:0,y:0,yaw:0}};
const a={id:"basket",pose:{x:1,y:0,yaw:0}},b={id:"washer",pose:{x:2,y:0,yaw:0}};
const input={cycle:{isRunning:true,pinnedSpot:source,pinnedMissionId:"laundry"},
 mission:{missionId:"laundry",mapName:"house"},mapName:"house",activeWaypointId:"washer",
 routeView:{orderedSpots:[a,b],executionSpots:[a,b],order:[{id:"basket",order:1},{id:"washer",order:2}]}};
test("child route includes observation departure and return, using the default mission palette",()=>{
 const overlay=recordMissionOverlay(input);
 expect(overlay.spots).toEqual([a,b]);
 expect(overlay.pathSpots).toEqual([source,a,b,source]);
 expect(overlay.activeWaypointId).toBe("washer");
 expect(overlay.color).toBeUndefined();
});
test("return retains route until cycle ends; selection alone cannot display it",()=>{
 expect(recordMissionOverlay({...input,cycle:{...input.cycle,returning:true}}).activeWaypointId).toBeNull();
 expect(recordMissionOverlay({...input,cycle:{...input.cycle,isRunning:false}})).toBeNull();
 expect(recordMissionOverlay({...input,cycle:{isRunning:false}})).toBeNull();
 expect(recordMissionOverlay({...input,mission:{missionId:"other",mapName:"house"}})).toBeNull();
 expect(recordMissionOverlay({...input,mapName:"other"})).toBeNull();
});
