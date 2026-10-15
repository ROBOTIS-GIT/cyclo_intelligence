import { render, screen, fireEvent } from '@testing-library/react';
import { useState } from 'react';
import ObjectLabelPicker from './ObjectLabelPicker';
import { collectionRequest } from '../../../utils/collectionApi';
jest.mock('../../../utils/collectionApi',()=>({collectionRequest:jest.fn()}));
function Editor(){
 const [value,setValue]=useState('');
 return <ObjectLabelPicker episodeId="ep" regions={[]} value={value} onChange={setValue}/>;
}
test('labels are selected or entered without a separate save request',async()=>{
 collectionRequest.mockResolvedValue({labels:['stick']});
 render(<Editor/>);
 fireEvent.click(await screen.findByRole('button',{name:'stick',exact:true}));
 expect(screen.getByRole('textbox',{name:'Object name'})).toHaveValue('stick');
 fireEvent.change(screen.getByRole('textbox',{name:'Object name'}),{target:{value:'box'}});
 expect(screen.getByRole('textbox',{name:'Object name'})).toHaveValue('box');
 expect(screen.queryByRole('button',{name:'Save label'})).not.toBeInTheDocument();
 expect(collectionRequest).toHaveBeenCalledTimes(1);
});
