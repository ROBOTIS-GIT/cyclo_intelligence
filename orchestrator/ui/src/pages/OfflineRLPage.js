// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';
import RLWorkflowLayout from '../features/offlineRL/components/RLWorkflowLayout';

export default function OfflineRLPage({ isActive = true }) {
  return <RLWorkflowLayout isActive={isActive} />;
}
