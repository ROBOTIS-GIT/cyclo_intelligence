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

import {createRobotAnchorStore} from "./robotAnchorStore";

test("retains equal snapshots but notifies a changed robot footprint",()=>{
 const store=createRobotAnchorStore(), listener=jest.fn();
 const unsubscribe=store.subscribe(listener);
 const anchor={x:200,y:100,width:800,height:600,robotBounds:{left:180,right:220,top:100,bottom:140}};
 store.set(anchor);store.set({...anchor,robotBounds:{...anchor.robotBounds}});
 expect(listener).toHaveBeenCalledTimes(1);
 expect(store.getSnapshot()).toBe(anchor);
 store.set({...anchor,robotBounds:{...anchor.robotBounds,left:160}});
 expect(listener).toHaveBeenCalledTimes(2);
 unsubscribe();store.set(null);expect(listener).toHaveBeenCalledTimes(2);
});
