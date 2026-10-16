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

import {fireEvent,render,screen} from "@testing-library/react";
import OperateDrawer from "./OperateDrawer";
import { autonomyRequest } from "../hooks/autonomyApi";
jest.mock("../hooks/autonomyApi", () => ({ autonomyRequest: jest.fn() }));
beforeEach(() => autonomyRequest.mockResolvedValue({messages:[],pending:false}));

test("details start folded, switch independently, close on Escape, and retain prompt drafts",()=>{
 const view=render(<OperateDrawer observation={{active:true}}/>);
 expect(screen.queryByRole('button',{name:'Status and settings'})).not.toBeInTheDocument();
 expect(screen.queryByRole('button',{name:'Activity'})).not.toBeInTheDocument();
 const message=screen.getByRole('button',{name:'Talk to robot'});
 fireEvent.click(message);
 expect(message).toHaveAttribute('aria-expanded','true');
 expect(message.closest('nav')).toHaveClass('operate-chat-trigger');
 fireEvent.change(screen.getByRole('textbox'),{target:{value:'Please wait here'}});
 fireEvent.click(screen.getByRole('button',{name:'Close details'}));
 expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
 fireEvent.click(message);expect(screen.getByRole('textbox')).toHaveValue('Please wait here');
 fireEvent.keyDown(window,{key:'Escape'});
 expect(message).toHaveAttribute('aria-expanded','false');
 expect(message).toHaveFocus();
 view.unmount();
});
