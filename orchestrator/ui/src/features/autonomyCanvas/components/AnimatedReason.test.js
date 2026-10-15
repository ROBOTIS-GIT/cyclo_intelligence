import {act,render,screen} from "@testing-library/react";
import AnimatedReason from "./AnimatedReason";
const result={decision:{decision:"start"},selected_mission:{mission_name:"CleanTable"}};
beforeEach(()=>jest.useFakeTimers());
afterEach(()=>jest.useRealTimers());
test("crossfades snapshots and preserves the outgoing action highlight",()=>{
 const view=render(<AnimatedReason text="I need to clean the table." result={result}/>);
 view.rerender(<AnimatedReason text="I need to wait." result={{decision:{decision:"wait"}}}/>);
 expect(view.container.querySelector(".is-outgoing strong")).toHaveTextContent("clean the table");
 expect(view.container.querySelector(".is-outgoing")).toHaveAttribute("aria-hidden","true");
 expect(view.container.querySelector(".is-current strong")).toHaveTextContent("wait");
 act(()=>jest.advanceTimersByTime(320));
 expect(view.container.querySelector(".is-outgoing")).toBeNull();
 expect(screen.getByText("I need to",{exact:false})).toBeInTheDocument();
});
test("empty polling results do not clear the current thought and identical text does not restart animation",()=>{
 const view=render(<AnimatedReason text="I need to wait." result={{decision:{decision:"wait"}}}/>);
 const current=view.container.querySelector(".is-current");
 view.rerender(<AnimatedReason text=""/>);
 expect(view.container.querySelector(".is-current")).toBe(current);
 expect(view.container.firstChild).toHaveAttribute("aria-busy","true");
 view.rerender(<AnimatedReason text="I need to wait." result={{decision:{decision:"wait"}}}/>);
 expect(view.container.querySelector(".is-current")).toBe(current);
 expect(view.container.querySelector(".is-outgoing")).toBeNull();
});
test("rapid updates settle on the newest thought; errors replace it immediately",()=>{
 const view=render(<AnimatedReason text="First"/>);
 view.rerender(<AnimatedReason text="Second"/>);
 act(()=>jest.advanceTimersByTime(100));
 view.rerender(<AnimatedReason text="Third"/>);
 act(()=>jest.advanceTimersByTime(320));
 expect(view.container.textContent).toBe("Third");
 view.rerender(<AnimatedReason text="Navigation failed" immediate/>);
 expect(view.container.textContent).toBe("Navigation failed");
 expect(view.container.querySelector(".is-outgoing")).toBeNull();
});
