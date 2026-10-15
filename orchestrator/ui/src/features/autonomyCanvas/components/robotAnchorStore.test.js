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
