import {render,screen} from '@testing-library/react';
import CollectAttentionPreview from './CollectAttentionPreview';
test('shows the heatmap paired with the current result and removes stale maps',()=>{
 const h=render(<CollectAttentionPreview result={{attention_preview:{image_base64:'abc',layer:23}}}/>);
 expect(screen.getByRole('img')).toHaveAttribute('src','data:image/jpeg;base64,abc');
 h.rerender(<CollectAttentionPreview result={{image_base64:'raw',attention_error:'unsupported'}}/>);
 expect(screen.getByRole('img')).toHaveAttribute('src','data:image/jpeg;base64,raw');
 expect(screen.getByRole('status')).toHaveTextContent('Attention unavailable');
});
