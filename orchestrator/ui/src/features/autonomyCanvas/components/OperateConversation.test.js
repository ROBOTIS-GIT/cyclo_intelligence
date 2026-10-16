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

import {fireEvent,render,screen,waitFor} from "@testing-library/react";
import OperateConversation from "./OperateConversation";
import {autonomyRequest} from "../hooks/autonomyApi";
jest.mock("../hooks/autonomyApi",()=>({autonomyRequest:jest.fn()}));
test("sends conversation messages and shows controller receipts",async()=>{
 autonomyRequest.mockImplementation((path,method,body)=>Promise.resolve(method==='POST'?{messages:[{id:'u',role:'user',content:body.text},{id:'r',role:'assistant',content:'Patrol direction is now reverse.'}],pending:false}:{messages:[],pending:false}));
 const view=render(<OperateConversation/>);
 await waitFor(()=>expect(autonomyRequest).toHaveBeenCalled());
 fireEvent.change(screen.getByRole('textbox'),{target:{value:'Reverse patrol'}});
 fireEvent.click(screen.getByRole('button',{name:'Send message'}));
 expect(await screen.findByText('Patrol direction is now reverse.')).toBeVisible();
 expect(autonomyRequest).toHaveBeenCalledWith('/autonomy/conversation','POST',expect.objectContaining({text:'Reverse patrol',request_id:expect.any(String)}));
 expect(screen.getByRole('textbox')).toHaveValue('');view.unmount();
});
