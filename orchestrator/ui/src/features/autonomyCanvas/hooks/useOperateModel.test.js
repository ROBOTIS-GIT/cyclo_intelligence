import { act, renderHook, waitFor } from '@testing-library/react';
import useOperateModel from './useOperateModel';
const model={id:'one',model_name:'Custom'};
beforeEach(()=>{localStorage.clear();global.fetch=jest.fn().mockResolvedValue({ok:true,json:async()=>({models:[model]})});});
afterEach(()=>{delete global.fetch;});
test('remembers selection per map and clears unavailable models after refresh',async()=>{
 const h=renderHook(({map})=>useOperateModel(map),{initialProps:{map:'map'}});
 await waitFor(()=>expect(h.result.current.models).toHaveLength(1));
 expect(h.result.current.selected).toBeNull();
 act(()=>h.result.current.select('one'));
 expect(localStorage.getItem('autonomy.operate.model.map')).toBe('one');
 h.rerender({map:'another'});
 expect(h.result.current.selected).toBeNull();
 await waitFor(()=>expect(h.result.current.loading).toBe(false));
 expect(h.result.current.selected).toBeNull();
 h.rerender({map:'map'});
 await waitFor(()=>expect(h.result.current.selected?.id).toBe('one'));
 fetch.mockResolvedValue({ok:true,json:async()=>({models:[]})});
 await act(async()=>h.result.current.refresh());
 expect(h.result.current.selected).toBeNull();h.unmount();
});
test('ignores a stale map response',async()=>{
 let finish;fetch.mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));
 const h=renderHook(({map})=>useOperateModel(map),{initialProps:{map:'old'}});
 h.rerender({map:'new'});
 await waitFor(()=>expect(h.result.current.models).toHaveLength(1));
 await act(async()=>finish({ok:true,json:async()=>({models:[{id:'old'}]})}));
 expect(h.result.current.models).toEqual([model]);h.unmount();
});
