import {fireEvent,render,screen} from "@testing-library/react";
import OperateDrawer from "./OperateDrawer";
import { autonomyRequest } from "../hooks/autonomyApi";
jest.mock("../hooks/autonomyApi", () => ({ autonomyRequest: jest.fn() }));
beforeEach(() => autonomyRequest.mockResolvedValue({messages:[],pending:false}));

test("details start folded, switch independently, close on Escape, and retain prompt drafts",()=>{
 const view=render(<OperateDrawer observation={{active:true}}/>);
 expect(screen.queryByText('Current situation')).not.toBeVisible();
 const message=screen.getByRole('button',{name:'Talk to robot'});
 fireEvent.click(message);
 expect(message).toHaveAttribute('aria-expanded','true');
 expect(message.closest('nav')).toHaveClass('operate-chat-trigger');
 fireEvent.change(screen.getByRole('textbox'),{target:{value:'Please wait here'}});
 fireEvent.click(screen.getByRole('button',{name:'Close details'}));
 expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
 fireEvent.click(message);expect(screen.getByRole('textbox')).toHaveValue('Please wait here');
 fireEvent.click(screen.getByRole('button',{name:'Activity'}));
 expect(screen.getByRole('list',{name:'Operation activity'})).toBeInTheDocument();
 expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
 fireEvent.keyDown(window,{key:'Escape'});
 expect(screen.getByRole('button',{name:'Activity'})).toHaveAttribute('aria-expanded','false');
 expect(screen.getByRole('button',{name:'Activity'})).toHaveFocus();
 view.unmount();
});
