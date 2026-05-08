// Copyright 2025 ROBOTIS CO., LTD.
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
// Author: Kiwoong Park

import React from 'react';
import clsx from 'clsx';

import { useSelector } from 'react-redux';

const classEpisodeStatusBody = clsx(
  'h-full',
  'w-full',
  'max-w-xs',
  'text-center',
  'flex',
  'flex-col',
  'items-center',
  'justify-around',
  'gap-1',
  'rounded-2xl',
  'border',
  'border-gray-200',
  'py-2',
  'px-3',
  'box-border',
  'shadow-md',
  'bg-white'
);

const TitleFontSize = 'clamp(1.5rem, 1.5vw, 2rem)';
const NumberFontSize = 'clamp(1.5rem, 1.5vw, 2rem)';

export default function EpisodeStatus() {
  const currentEpisodeNumber = useSelector((state) => state.tasks.recordStatus.currentEpisodeNumber);
  const numEpisodes = useSelector((state) => state.tasks.taskInfo.numEpisodes);

  return (
    <div className={classEpisodeStatusBody}>
      <div
        className="w-full h-full flex justify-center items-center"
        style={{ fontSize: TitleFontSize }}
      >
        Episode
      </div>
      <div
        className="w-full h-full flex justify-center items-center bg-gray-200 rounded-lg px-3 font-bold whitespace-nowrap"
        style={{ fontSize: NumberFontSize }}
      >
        <span className="font-bold">{currentEpisodeNumber}</span>
        <span className="text-gray-600">{' / '}</span>
        <span className="text-gray-600">{numEpisodes}</span>
      </div>
    </div>
  );
}
