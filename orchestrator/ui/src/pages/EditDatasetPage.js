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

import React, { useEffect, useState } from 'react';
import clsx from 'clsx';
import toast, { useToasterStore } from 'react-hot-toast';
import {
  MdCloudUpload,
  MdDeleteSweep,
  MdMerge,
  MdMovie,
  MdPlayCircle,
} from 'react-icons/md';
import HuggingfaceSection from '../features/editDataset/components/DatasetHuggingfaceSection';
import MergeSection from '../features/editDataset/components/DatasetMergeSection';
import DeleteSection from '../features/editDataset/components/DatasetDeleteSection';
import ConvertSection from '../features/editDataset/components/DatasetConvertSection';
import ReplayPage from './ReplayPage';

const TOAST_LIMIT = 3;

const SECTION_TYPES = {
  REVIEW: 'review',
  DELETE: 'delete',
  MERGE: 'merge',
  CONVERT: 'convert',
  HUGGINGFACE: 'huggingface',
};

const SECTION_CONFIG = {
  [SECTION_TYPES.REVIEW]: {
    label: 'Review Episodes',
    icon: MdPlayCircle,
  },
  [SECTION_TYPES.DELETE]: {
    label: 'Delete Episodes',
    icon: MdDeleteSweep,
  },
  [SECTION_TYPES.MERGE]: {
    label: 'Merge Rosbag Datasets',
    icon: MdMerge,
  },
  [SECTION_TYPES.CONVERT]: {
    label: 'Convert Dataset',
    icon: MdMovie,
  },
  [SECTION_TYPES.HUGGINGFACE]: {
    label: 'Hugging Face Upload & Download',
    icon: MdCloudUpload,
  },
};

const SECTION_ORDER = [
  SECTION_TYPES.REVIEW,
  SECTION_TYPES.DELETE,
  SECTION_TYPES.MERGE,
  SECTION_TYPES.CONVERT,
  SECTION_TYPES.HUGGINGFACE,
];

const manageToastLimit = (toasts) => {
  toasts
    .filter((t) => t.visible)
    .filter((_, i) => i >= TOAST_LIMIT)
    .forEach((t) => toast.dismiss(t.id));
};

export default function EditDatasetPage() {
  const { toasts } = useToasterStore();
  const [activeSection, setActiveSection] = useState(SECTION_TYPES.REVIEW);

  useEffect(() => {
    manageToastLimit(toasts);
  }, [toasts]);

  const renderActiveSection = () => {
    switch (activeSection) {
      case SECTION_TYPES.REVIEW:
        return <ReplayPage isActive />;
      case SECTION_TYPES.DELETE:
        return <DeleteSection isEditable />;
      case SECTION_TYPES.MERGE:
        return <MergeSection isEditable />;
      case SECTION_TYPES.CONVERT:
        return <ConvertSection isEditable />;
      case SECTION_TYPES.HUGGINGFACE:
        return <HuggingfaceSection isEditable />;
      default:
        return <ReplayPage isActive />;
    }
  };

  return (
    <div className="flex h-full min-h-0 w-full flex-col overflow-hidden bg-gray-50">
      <div className="flex h-16 flex-shrink-0 items-center justify-center border-b border-gray-200 bg-white px-4">
        <div className="flex max-w-full items-center justify-center gap-1 overflow-x-auto rounded-lg border border-gray-200 bg-gray-100 p-1 shadow-sm">
          {SECTION_ORDER.map((sectionType) => {
            const config = SECTION_CONFIG[sectionType];
            const Icon = config.icon;
            const isActive = activeSection === sectionType;

            return (
              <button
                key={sectionType}
                type="button"
                onClick={() => setActiveSection(sectionType)}
                className={clsx(
                  'flex h-10 shrink-0 items-center gap-2 rounded-md border px-3 text-sm font-semibold transition-colors',
                  'focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 focus-visible:ring-offset-1',
                  isActive
                    ? 'border-blue-200 bg-white text-blue-700 shadow-sm'
                    : 'border-transparent text-gray-600 hover:bg-white hover:text-gray-900 hover:shadow-sm'
                )}
              >
                <Icon
                  size={18}
                  className={isActive ? 'text-blue-600' : 'text-gray-500'}
                />
                <span className="whitespace-nowrap">{config.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div className={clsx(
        'min-h-0 flex-1 overflow-hidden',
        activeSection === SECTION_TYPES.REVIEW ? 'bg-gray-50' : 'overflow-y-auto p-6'
      )}>
        {renderActiveSection()}
      </div>
    </div>
  );
}
