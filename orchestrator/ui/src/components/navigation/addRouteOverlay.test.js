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

import * as THREE from "three";
import { addRouteOverlay } from "./addRouteOverlay";
test("mission overlay is colored but cannot intercept Observation clicks or drags",()=>{
 const layer=new THREE.Group(), make=()=>{const group=new THREE.Group();group.userData={spotId:"same",waypointHitAnchor:true};const child=new THREE.Group();child.userData={spotId:"same",pulse:true,pulseBase:.5};group.add(child);return group;};
 const marker=jest.fn(make),badge=jest.fn(make),line=jest.fn(make);
 const spot={id:"same",pose:{x:1,y:2}};
 addRouteOverlay(layer,{spots:[spot],pathSpots:[spot,spot],order:[{id:"same",order:1}],activeWaypointId:"same",color:0x8d7899,activeColor:0x655174},1,{}, {marker,badge,line});
 expect(marker).toHaveBeenCalledWith(spot,false,1,true,{idle:0x8d7899,selected:0x655174});
 expect(line).toHaveBeenCalledWith(expect.any(Array),0x8d7899,1);
 expect(layer.children).toHaveLength(3);
 layer.children.forEach(group=>group.traverse(child=>{
  expect(child.userData.spotId).toBeUndefined();expect(child.userData.waypointHitAnchor).toBeUndefined();
  const hits=[];child.raycast({},hits);expect(hits).toEqual([]);
 }));
 expect(layer.children[1].children[0].userData.pulse).toBe(true);
});
