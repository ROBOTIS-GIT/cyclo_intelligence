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

import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { setSelectedNodeId } from '../../features/actionCanvas/actionCanvasSlice';
import BtParameterPanel from '../../features/btEditor/parameters/BtParameterPanel';

export default function BTParamPanel(props) {
  const dispatch = useDispatch();
  const robotType = useSelector((state) => state.tasks?.robotType || '');
  return <BtParameterPanel {...props} robotType={robotType}
    onClose={props.onClose || (() => dispatch(setSelectedNodeId(null)))} />;
}
