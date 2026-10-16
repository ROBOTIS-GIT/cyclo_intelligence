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

import {act,fireEvent,render,screen} from "@testing-library/react";
import OperateRobotThought from "./OperateRobotThought";
import {thoughtPlacement} from "./robotThought";
jest.mock("react-redux", () => ({useSelector: fn => fn({ros:{rosHost:"robot.local"}})}));

const state={active:true,phase:"observing",execution:{lifecycle:"active",activity:"observing"},
 context_version:4,result:{context_version:4,reasoning_task:"mission_selection",camera_topic:"/head/image/compressed",
 decision:{decision:"start",reason:"Empty dishes remain on the table."}}};
const anchor={x:500,y:500,width:1000,height:700};
beforeEach(()=>jest.useFakeTimers());
afterEach(()=>jest.useRealTimers());

test("shows a single live stream, actual action, and reason without model captions",()=>{
 const {container}=render(<OperateRobotThought observation={state} anchor={anchor}/>);
 expect(screen.getByText("Empty dishes remain on the table.")).toBeInTheDocument();
 const image=screen.getByAltText("Live head camera");
 expect(image.src).toContain("/stream?type=ros_compressed");
 expect(image.src).toContain("topic=/head/image");
 expect(screen.getAllByRole("img")).toHaveLength(1);
 expect(container.querySelector(".operate-thought-camera")).not.toHaveClass("is-visible");
 fireEvent.load(image);
 expect(container.querySelector(".operate-thought-camera")).not.toHaveClass("is-visible");
 act(()=>jest.advanceTimersByTime(40));
 expect(container.querySelector(".operate-thought-camera")).toHaveClass("is-visible");
 expect(container.querySelector(".operate-thought-bubble")).toHaveStyle({left:"16px",top:"204px"});
 expect(screen.getByRole("status",{name:"Current action"})).toHaveTextContent("Looking around");
 expect(container.querySelector("figcaption")).toBeNull();
 const bubble=container.querySelector(".operate-thought-content");
 expect([...bubble.children].indexOf(bubble.querySelector(".operate-thought-camera"))).toBeLessThan([...bubble.children].indexOf(bubble.querySelector(".operate-thought-reason-stack")));
 expect(screen.queryByText(/Heading back|Robot view|Connecting camera|Qwen|LoRA/)).not.toBeInTheDocument();
});

test("awaiting a fresh context retains the last thought without reconnecting the camera",()=>{
 const view=render(<OperateRobotThought observation={state} anchor={anchor}/>);
 const image=screen.getByAltText("Live head camera");
 view.rerender(<OperateRobotThought observation={{...state,context_version:5,result:null}} anchor={anchor}/>);
 expect(screen.getByText("Empty dishes remain on the table.")).toBeInTheDocument();
 expect(view.container.querySelector(".operate-thought-reason-stack")).toHaveAttribute("aria-busy","true");
 expect(screen.getByAltText("Live head camera")).toBe(image);
 expect(image.src).toContain("topic=/head/image");
});

test("start connects, pause keeps reasoning visible, and Stop fades before disconnecting",()=>{
 const view=render(<OperateRobotThought observation={{phase:"idle"}} anchor={anchor}/>);
 expect(screen.queryByAltText("Live head camera")).not.toBeInTheDocument();
 view.rerender(<OperateRobotThought observation={state} anchor={anchor}/>);
 const image=screen.getByAltText("Live head camera");fireEvent.load(image);
 act(()=>jest.advanceTimersByTime(40));
 view.rerender(<OperateRobotThought observation={{...state,phase:"paused",paused:true}} anchor={anchor}/>);
 expect(view.container.querySelector(".operate-robot-thought")).toHaveClass("is-visible");
 expect(screen.getByAltText("Live head camera")).toBe(image);
 view.rerender(<OperateRobotThought observation={{...state,phase:"stopping",execution:{lifecycle:"stopping"}}} anchor={anchor}/>);
 expect(view.container.querySelector(".operate-robot-thought")).not.toHaveClass("is-visible");
 expect(view.container.querySelector(".operate-thought-camera")).not.toHaveClass("is-visible");
 expect(image).toHaveAttribute("src");
 act(()=>jest.advanceTimersByTime(699));
 expect(image).toHaveAttribute("src");
 act(()=>jest.advanceTimersByTime(1));
 expect(image).not.toHaveAttribute("src");
 expect(screen.queryByAltText("Live head camera")).not.toBeInTheDocument();
});

test("rapid restart cancels pending camera removal and unmount always disconnects",()=>{
 const view=render(<OperateRobotThought observation={state} anchor={anchor}/>);
 const image=screen.getByAltText("Live head camera");fireEvent.load(image);
 act(()=>jest.advanceTimersByTime(40));
 view.rerender(<OperateRobotThought observation={{...state,execution:{lifecycle:"stopped"}}} anchor={anchor}/>);
 act(()=>jest.advanceTimersByTime(100));
 view.rerender(<OperateRobotThought observation={state} anchor={anchor}/>);
 act(()=>jest.advanceTimersByTime(800));
 expect(screen.getByAltText("Live head camera")).toBe(image);
 view.unmount();expect(image).not.toHaveAttribute("src");
});

test("failed sessions retain the actual error without an inactive camera",()=>{
 render(<OperateRobotThought observation={{...state,error:"Navigation failed",execution:{lifecycle:"failed"}}} anchor={anchor}/>);
 expect(screen.getByText("Navigation failed")).toBeInTheDocument();
 expect(screen.queryByAltText("Live head camera")).not.toBeInTheDocument();
 expect(screen.queryByText("Empty dishes remain on the table.")).not.toBeInTheDocument();
});

test("unknown location docks silently; viewport edges never push the tail over the robot",()=>{
 const {container}=render(<OperateRobotThought observation={state}/>);
 expect(container.querySelector(".operate-thought-bubble")).toHaveClass("is-docked");
 expect(container.querySelector(".operate-thought-connector")).toBeNull();
 expect(screen.queryByText(/outside the map view/)).not.toBeInTheDocument();
 const position=thoughtPlacement(anchor,300,250);
 expect(position.x+150).toBe(anchor.x);expect(position.y+250).toBeLessThan(anchor.y);
 const edge=thoughtPlacement({...anchor,x:950,y:20},300,250);
 expect(edge.x+edge.width).toBeLessThanOrEqual(920);expect(edge.y).toBeGreaterThanOrEqual(0);
 expect(edge.side).toBe("left");
 expect(thoughtPlacement(null,300,250)).toBeNull();
});


test("action emphasis is preserved during Stop fade and omitted from errors",()=>{
 const observation={...state,result:{...state.result,selected_mission:{mission_name:"CleanTable"},
  decision:{decision:"start",reason:"I need to clean the table."}}};
 const view=render(<OperateRobotThought observation={observation} anchor={anchor}/>);
 expect(view.container.querySelector("strong.operate-thought-action")).toHaveTextContent("clean the table");
 view.rerender(<OperateRobotThought observation={{...observation,phase:"stopping"}} anchor={anchor}/>);
 expect(view.container.querySelector("strong.operate-thought-action")).toHaveTextContent("clean the table");
 view.rerender(<OperateRobotThought observation={{...observation,phase:"error",error:"CleanTable failed"}} anchor={anchor}/>);
 expect(view.container.querySelector("strong.operate-thought-action")).toBeNull();
});


test("a fresh execution session does not display the previous session's thought",()=>{
 const view=render(<OperateRobotThought observation={{...state,session_id:"first"}} anchor={anchor}/>);
 expect(screen.getByText("Empty dishes remain on the table.")).toBeInTheDocument();
 view.rerender(<OperateRobotThought observation={{...state,session_id:"second",result:null}} anchor={anchor}/>);
 expect(screen.queryByText("Empty dishes remain on the table.")).not.toBeInTheDocument();
});


test("cloud tail has three circles growing towards the bubble even near the top edge",()=>{
 const view=render(<OperateRobotThought observation={state} anchor={{...anchor,y:100}}/>);
 const circles=[...view.container.querySelectorAll(".operate-thought-connector circle")];
 expect(circles.map(circle=>Number(circle.getAttribute("r")))).toEqual([11,7,4]);
 expect(circles[0].style.transform).not.toBe(circles[2].style.transform);
 expect(view.container.querySelector(".operate-thought-content")).toContainElement(screen.getByAltText("Live head camera"));
});

test("layout stays inside the viewport and outside the whole rotated robot silhouette",()=>{
 for (const width of [700,1000,1600]) for (const height of [500,900]) {
  for (const x of [30,width/2,width-30]) for (const y of [30,height/2,height-60]) {
   const robotBounds={left:x-30,right:x+30,top:y,bottom:y+50};
   const position=thoughtPlacement({x,y,width,height,robotBounds},480,480);
   expect(position.x).toBeGreaterThanOrEqual(0);
   expect(position.y).toBeGreaterThanOrEqual(0);
   expect(position.x+position.width).toBeLessThanOrEqual(width);
   expect(position.y+position.height).toBeLessThanOrEqual(height);
   if(position.side!=="docked") {
    expect(position.x+position.width<=robotBounds.left || position.x>=robotBounds.right ||
     position.y+position.height<=robotBounds.top || position.y>=robotBounds.bottom).toBe(true);
    for(const circle of position.circles) {
     expect(circle.x+circle.r<=robotBounds.left || circle.x-circle.r>=robotBounds.right ||
      circle.y+circle.r<=robotBounds.top || circle.y-circle.r>=robotBounds.bottom).toBe(true);
    }
   }
  }
 }
});

test("prefers above when it fits, retaining a suitable side across small pose changes",()=>{
 const roomy={...anchor,width:1600,height:1000,x:800,y:800};
 expect(thoughtPlacement(roomy,480,400).side).toBe("above");
 const initial=thoughtPlacement(anchor,480,480);
 const moved=thoughtPlacement({...anchor,x:anchor.x+2,y:anchor.y+2},480,480,initial.side);
 expect(moved.side).toBe(initial.side);
 expect(thoughtPlacement({...anchor,width:220,height:190},480,480).side).toBe("docked");
});

test("Stop before the first camera paint cancels pending fade-in",()=>{
 const view=render(<OperateRobotThought observation={state} anchor={anchor}/>);
 fireEvent.load(screen.getByAltText("Live head camera"));
 view.rerender(<OperateRobotThought observation={{phase:"stopped"}} anchor={anchor}/>);
 act(()=>jest.advanceTimersByTime(80));
 expect(view.container.querySelector(".operate-robot-thought")).not.toHaveClass("is-visible");
 expect(view.container.querySelector(".operate-thought-camera")).not.toHaveClass("is-visible");
});
