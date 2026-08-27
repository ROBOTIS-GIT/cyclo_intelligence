// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';

/**
 * Compact line icon inspired by a humanoid robot balancing on one leg.
 *
 * The paths use currentColor so the same mark stays legible in the global
 * navigation, the PLAYGROUND rail, and the workspace status control.
 */
export default function RobotLabIcon({ size = 24, className = '', ...props }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.65"
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      data-robot-lab-icon="true"
      {...props}
    >
      <path d="M9.6 4.2c.45-1.35 1.28-2.05 2.48-2.05 1.2 0 2.03.7 2.48 2.05" />
      <path d="M9.15 4.2h5.85v2.25c0 1.65-1.18 2.8-2.92 2.8-1.75 0-2.93-1.15-2.93-2.8V4.2Z" />
      <path d="M9.35 5.7h5.45" />
      <path d="M10.15 9.05 9.1 14.2l3.15 1.25 3.05-1.8-.95-4.6" />
      <circle cx="8.85" cy="10.15" r=".72" />
      <circle cx="15.05" cy="9.55" r=".72" />
      <path d="m8.15 10.35-4.2 1.3-2.35-.35" />
      <path d="m15.75 9.35 3.75-1.8 2.9.25" />
      <circle cx="3.8" cy="11.65" r=".58" />
      <circle cx="19.55" cy="7.55" r=".58" />
      <path d="m11.75 15.25-.55 3.1-.25 3.25" />
      <circle cx="11.2" cy="18.3" r=".65" />
      <path d="m10.95 21.6-1.55.25" />
      <path d="m12.75 15.15 3.05 1.45 2.75 1.05" />
      <circle cx="15.8" cy="16.6" r=".65" />
      <path d="m18.55 17.65 1.25 1.1" />
    </svg>
  );
}
