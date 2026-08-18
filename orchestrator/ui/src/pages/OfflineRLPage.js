// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React, { useState } from 'react';
import DatasetConvertSection from '../features/editDataset/components/DatasetConvertSection';
import OfflineRLTrainingSection from '../features/offlineRL/components/OfflineRLTrainingSection';

export default function OfflineRLPage({ isActive = true }) {
  const [trainingRunning, setTrainingRunning] = useState(true);
  const [datasetBusy, setDatasetBusy] = useState(false);

  return (
    <div className="h-full min-h-0 w-full overflow-y-auto bg-gray-50 p-6">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6">
        <DatasetConvertSection
          isEditable={isActive && !trainingRunning}
          onBusyChange={setDatasetBusy}
        />
        <OfflineRLTrainingSection
          isActive={isActive && !datasetBusy}
          onRunningChange={setTrainingRunning}
        />
      </div>
    </div>
  );
}
