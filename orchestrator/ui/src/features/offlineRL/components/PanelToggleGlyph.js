// Copyright 2025 ROBOTIS CO., LTD.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.

import React from 'react';

export default function PanelToggleGlyph({
  glyphTestId,
  accentTestId,
  accentSide = 'left',
}) {
  const accentClassName = accentSide === 'bottom'
    ? 'inset-x-0 bottom-0 h-[20%]'
    : `inset-y-0 w-[20%] ${accentSide === 'right' ? 'right-0' : 'left-0'}`;

  return (
    <span
      data-testid={glyphTestId}
      aria-hidden="true"
      className="relative block h-[17px] w-[22px] overflow-hidden rounded-[5px] border border-current bg-[#f3f0e8]"
    >
      <span
        data-testid={accentTestId}
        className={`absolute bg-[#627d68] ${accentClassName}`}
      />
    </span>
  );
}
