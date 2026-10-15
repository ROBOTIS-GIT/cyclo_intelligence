import {render,screen,fireEvent,waitFor} from '@testing-library/react';
import ReasoningModelDelete from './ReasoningModelDelete';
import {collectionRequest} from '../../../utils/collectionApi';
jest.mock('../../../utils/collectionApi',()=>({collectionRequest:jest.fn()}));
const model={id:'saved',model_name:'house test'};
beforeEach(()=>{jest.clearAllMocks();jest.spyOn(window,'confirm').mockReturnValue(true);});
afterEach(()=>jest.restoreAllMocks());
test('confirms then deletes and refreshes',async()=>{
 const refresh=jest.fn();collectionRequest.mockResolvedValue({});
 render(<ReasoningModelDelete model={model} onDeleted={refresh}/>);
 fireEvent.click(screen.getByRole('button'));
 await waitFor(()=>expect(refresh).toHaveBeenCalledTimes(1));
 expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('house test'));
 expect(collectionRequest).toHaveBeenCalledWith('qwen/models/saved',undefined,'DELETE');
});
test('cancel leaves model untouched',()=>{
 window.confirm.mockReturnValue(false);render(<ReasoningModelDelete model={model}/>);
 fireEvent.click(screen.getByRole('button'));expect(collectionRequest).not.toHaveBeenCalled();
});
test('base models cannot be deleted',()=>{
 render(<ReasoningModelDelete model={{...model,id:'base:qwen'}}/>);
 expect(screen.getByRole('button')).toBeDisabled();
});
test('in-use rejection remains visible without refreshing',async()=>{
 const refresh=jest.fn();collectionRequest.mockRejectedValue(new Error('Stop observation first'));
 render(<ReasoningModelDelete model={model} onDeleted={refresh}/>);
 fireEvent.click(screen.getByRole('button'));
 expect(await screen.findByRole('alert')).toHaveTextContent('Stop observation first');
 expect(refresh).not.toHaveBeenCalled();
});
