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
// Author: Dongyun Kim

import { useEffect, useRef } from 'react';

/**
 * Hook for handling keyboard shortcuts in Replay Viewer.
 * Uses refs to avoid stale closures in event handlers.
 *
 * @param {Object} params - Hook parameters
 * @param {boolean} params.isActive - Whether the page is active
 * @param {Object} params.handlers - Object containing handler functions
 */
export function useKeyboardShortcuts({
    isActive,
    handlers,
}) {
    // Refs to hold latest handler functions
    const handlersRef = useRef(handlers);

    // Update refs when handlers change
    useEffect(() => {
        handlersRef.current = handlers;
    }, [handlers]);

    useEffect(() => {
        if (!isActive) return;

        const handleKeyDown = (e) => {
            // Ignore when focused on input fields
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

            const h = handlersRef.current;

            // Using e.code for IME compatibility (works with Korean/Japanese input mode)
            switch (e.code) {
                // Navigation
                case 'ArrowUp':
                    e.preventDefault();
                    h.onNavigatePrev?.();
                    break;
                case 'ArrowDown':
                    e.preventDefault();
                    h.onNavigateNext?.();
                    break;

                // Seeking
                case 'ArrowLeft':
                    e.preventDefault();
                    if (e.shiftKey) {
                        h.onSeekRelative?.(-5);
                    } else {
                        h.onStepBackward?.();
                    }
                    break;
                case 'ArrowRight':
                    e.preventDefault();
                    if (e.shiftKey) {
                        h.onSeekRelative?.(5);
                    } else {
                        h.onStepForward?.();
                    }
                    break;

                // Playback
                case 'Space':
                    e.preventDefault();
                    h.onTogglePlayPause?.();
                    break;

                default:
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isActive]);

    return null;
}

/**
 * Default keyboard shortcuts help content
 */
export const KEYBOARD_SHORTCUTS = [
    { key: 'Space', description: 'Play/Pause' },
    { key: '←/→', description: 'Step frame backward/forward' },
    { key: 'Shift + ←/→', description: 'Seek ±5 seconds' },
    { key: '↑/↓', description: 'Previous/Next rosbag' },
];

export default useKeyboardShortcuts;
