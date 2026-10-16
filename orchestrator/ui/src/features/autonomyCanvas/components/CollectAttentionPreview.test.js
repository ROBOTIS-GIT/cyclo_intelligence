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

import {render,screen} from '@testing-library/react';
import CollectAttentionPreview from './CollectAttentionPreview';
test('shows the heatmap paired with the current result and removes stale maps',()=>{
 const h=render(<CollectAttentionPreview result={{attention_preview:{image_base64:'abc',layer:23}}}/>);
 expect(screen.getByRole('img')).toHaveAttribute('src','data:image/jpeg;base64,abc');
 h.rerender(<CollectAttentionPreview result={{image_base64:'raw',attention_error:'unsupported'}}/>);
 expect(screen.getByRole('img')).toHaveAttribute('src','data:image/jpeg;base64,raw');
 expect(screen.getByRole('status')).toHaveTextContent('Attention unavailable');
});
