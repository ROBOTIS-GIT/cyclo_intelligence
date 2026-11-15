import React, { useId } from 'react';

const COLORS = {
  success: ['#54715c', '#96b19b', '#486650'],
  failure: ['#9e615a', '#dfaaa0', '#955c54'],
  unlabeled: ['#8e887e', '#d1cbc0', '#80796f'],
};

/** One SVG vessel for collection, selected datasets and training.
 * Geometry stays in one coordinate system; at most 36 plates are drawn.
 * Capacity is a display reference, never a dataset limit.
 */
export default function ReplayBufferVessel({
  counts, capacity = 200, label = 'Replay buffer composition', outcomeProps = null,
}) {
  const id = `vessel-${useId().replaceAll(':', '')}`;
  const entries = Object.keys(COLORS).map((outcome) => ({
    outcome, count: Math.max(0, Math.floor(Number(counts[outcome]) || 0)),
  }));
  const total = entries.reduce((sum, item) => sum + item.count, 0);
  const reference = Math.max(1, capacity);
  const occupied = Math.min(total, reference);
  const discCount = Math.min(total, 36);
  const visible = entries.filter((item) => item.count > 0);
  // Reserve one plate per label, including small minorities in large buffers.
  const remaining = Math.max(0, discCount - visible.length);
  visible.forEach((item) => {
    item.plates = 1 + Math.floor(remaining * item.count / total);
  });
  let extra = discCount - visible.reduce((sum, item) => sum + item.plates, 0);
  for (let i = 0; extra > 0; i += 1, extra -= 1) visible[i % visible.length].plates += 1;
  let bottom = 164;
  const stackHeight = 128 * occupied / reference;
  const stacks = visible.map((item) => {
    const height = stackHeight * item.count / total;
    const step = height / item.plates;
    const plates = Array.from({ length: item.plates }, (_, index) => ({
      bottom: bottom - index * step,
      top: bottom - index * step - step * 0.88,
    }));
    bottom -= height;
    return { ...item, plates };
  });
  const interactive = Boolean(outcomeProps);
  return (
    <svg
      viewBox="0 0 180 204"
      className="pg-buffer-vessel min-w-0 max-w-full shrink"
      role={interactive ? 'group' : 'img'}
      aria-label={label}
      data-capacity-used={occupied}
      data-capacity-empty={Math.max(0, reference - occupied)}
      data-capacity-percent={Math.round(100 * occupied / reference)}
      data-visible-disc-count={discCount}
    >
      <defs>
        <linearGradient id={`${id}-shell`}>
          <stop offset="0" stopColor="#b8b2a5" stopOpacity=".5" />
          <stop offset=".26" stopColor="#fffdf8" stopOpacity=".25" />
          <stop offset=".7" stopColor="#ece7dc" stopOpacity=".12" />
          <stop offset="1" stopColor="#aaa394" stopOpacity=".45" />
        </linearGradient>
        {Object.entries(COLORS).map(([outcome, colors]) => (
          <linearGradient key={outcome} id={`${id}-${outcome}`}>
            {colors.map((color, index) => <stop key={color} offset={index / 2} stopColor={color} />)}
          </linearGradient>
        ))}
        <radialGradient id={`${id}-shadow`}>
          <stop stopColor="#655c4e" stopOpacity=".2" />
          <stop offset="1" stopColor="#655c4e" stopOpacity="0" />
        </radialGradient>
      </defs>
      <ellipse cx="90" cy="189" rx="84" ry="13" fill={`url(#${id}-shadow)`} />
      {/* Low tray: rear lip, inset floor, then a front rim after the stack. */}
      <path d="M12 168 A78 12 0 0 1 168 168 L168 175 A78 12 0 0 1 12 175 Z" fill="#d4cec1" stroke="#b4ad9f" />
      <ellipse cx="90" cy="168" rx="78" ry="12" fill="#ebe6dc" stroke="#bcb4a6" />
      <rect x="20" y="24" width="140" height="140" fill={`url(#${id}-shell)`} data-testid="replay-cylinder-empty-capacity" />
      <ellipse cx="90" cy="164" rx="70" ry="10" fill="#ded8cb" stroke="#b8af9f" data-testid="replay-cylinder-base" />
      <g
        data-testid="replay-cylinder-occupied-capacity" data-rendered-discs={discCount}
        data-full-plate-stack="true" data-center-x="90" data-radius-x="70" data-base-y="164"
      >
        {stacks.map(({ outcome, count, plates }, stackIndex) => {
          const props = outcomeProps?.(outcome, count) || {};
          return (
            <g
              key={outcome} {...props} data-episode-count={count}
              role={interactive ? 'button' : undefined}
              tabIndex={interactive ? 0 : undefined}
              className={interactive ? 'cursor-pointer outline-none hover:brightness-110 focus-visible:brightness-125' : undefined}
              onKeyDown={interactive ? (event) => {
                if (event.key === 'Enter' || event.key === ' ') {
                  event.preventDefault();
                  props.onClick?.(event);
                }
              } : undefined}
            >
              {plates.map((plate, index) => (
                <g key={index} data-testid="replay-cylinder-disc" data-outcome={outcome}
                  data-center-x="90" data-radius-x="70" data-bottom-y={plate.bottom}
                  data-base-aligned={stackIndex === 0 && index === 0 ? 'true' : undefined}>
                  <path
                    d={`M20 ${plate.top} L20 ${plate.bottom} A70 10 0 0 0 160 ${plate.bottom} L160 ${plate.top} Z`}
                    fill={`url(#${id}-${outcome})`} stroke={COLORS[outcome][0]} strokeWidth=".6"
                    data-testid="replay-cylinder-disc-edge"
                  />
                  <ellipse cx="90" cy={plate.top} rx="70" ry="10"
                    fill={`url(#${id}-${outcome})`} stroke={COLORS[outcome][1]} strokeWidth=".75"
                    data-testid="replay-cylinder-disc-face" />
                </g>
              ))}
            </g>
          );
        })}
      </g>
      <g pointerEvents="none">
        <path d="M20 24 V164 M160 24 V164" stroke="#b8af9f" strokeWidth=".8" />
        <path d="M30 36 V146" stroke="#fffef8" strokeOpacity=".55" strokeWidth="3" strokeLinecap="round" />
        <ellipse cx="90" cy="24" rx="70" ry="10" fill="none" stroke="#b4aa99" strokeWidth="1.3" />
        <ellipse cx="90" cy="24" rx="65" ry="7" fill="none" stroke="#fffdf7" strokeOpacity=".8" />
        <path d="M12 168 A78 12 0 0 0 168 168 L168 175 A78 12 0 0 1 12 175 Z"
          fill="#d7d0c2" stroke="#b1a796" strokeWidth=".85" data-testid="replay-cylinder-base-front" />
        <path d="M15 168 A75 11 0 0 0 165 168" fill="none" stroke="#fffdf6" strokeWidth="1.4" />
      </g>
    </svg>
  );
}
