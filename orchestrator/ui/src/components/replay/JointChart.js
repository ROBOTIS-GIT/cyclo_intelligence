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

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { MdExpandMore, MdExpandLess } from 'react-icons/md';
import {
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    ReferenceLine,
    ReferenceDot,
} from 'recharts';

// Fixed colors for state and action
const STATE_COLOR = '#dc2626';  // Red for state
const ACTION_COLOR = '#2563eb'; // Blue for action

const closestPointForTime = (data, time) => {
    if (!data.length) return null;
    let low = 0;
    let high = data.length - 1;
    const target = Number(time) || 0;

    while (low <= high) {
        const mid = (low + high) >> 1;
        const midTime = Number(data[mid]?.time) || 0;
        if (midTime < target) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    if (low <= 0) return data[0];
    if (low >= data.length) return data[data.length - 1];

    const before = data[low - 1];
    const after = data[low];
    return Math.abs((after.time || 0) - target) < Math.abs(target - (before.time || 0))
        ? after
        : before;
};

const valueForSeriesAtTime = (data, time, key) => {
    const closest = closestPointForTime(data, time);
    const value = closest?.[key];
    return typeof value === 'number' ? value : null;
};

export const buildJointChartData = (stateData, actionData, name, hasAction = false) => {
    const stateKey = `state_${name}`;
    const actionKey = `action_${name}`;
    const times = new Set();

    stateData.forEach((point) => {
        if (typeof point?.[stateKey] === 'number') {
            times.add(point.time);
        }
    });
    if (hasAction) {
        actionData.forEach((point) => {
            if (typeof point?.[actionKey] === 'number') {
                times.add(point.time);
            }
        });
    }

    return Array.from(times)
        .filter((time) => typeof time === 'number' && Number.isFinite(time))
        .sort((a, b) => a - b)
        .map((time) => {
            const point = { time };
            const stateValue = valueForSeriesAtTime(stateData, time, stateKey);
            if (stateValue !== null) {
                point[stateKey] = stateValue;
            }
            if (hasAction) {
                const actionValue = valueForSeriesAtTime(actionData, time, actionKey);
                if (actionValue !== null) {
                    point[actionKey] = actionValue;
                }
            }
            return point;
        });
};

const JointTooltip = ({ active, label, stateData, actionData, name, hasAction }) => {
    if (!active || label === undefined || label === null) return null;

    const stateValue = valueForSeriesAtTime(stateData, label, `state_${name}`);
    const actionValue = hasAction
        ? valueForSeriesAtTime(actionData, label, `action_${name}`)
        : null;

    return (
        <div className="rounded border border-gray-200 bg-white/95 px-2 py-1.5 text-[11px] shadow-sm">
            <div className="mb-1 font-medium text-gray-700">
                Time: {Number(label).toFixed(2)}s
            </div>
            <div className="flex items-center gap-1.5 text-gray-700">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: STATE_COLOR }} />
                <span className="w-10">State</span>
                <span className="font-mono">
                    {stateValue !== null ? stateValue.toFixed(4) : '-'}
                </span>
            </div>
            {hasAction && (
                <div className="mt-0.5 flex items-center gap-1.5 text-gray-700">
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: ACTION_COLOR }} />
                    <span className="w-10">Action</span>
                    <span className="font-mono">
                        {actionValue !== null ? actionValue.toFixed(4) : '-'}
                    </span>
                </div>
            )}
        </div>
    );
};

/**
 * Individual Joint Chart Component showing both State and Action.
 *
 * @param {Object} props
 * @param {string} props.name - Joint name
 * @param {Array} props.stateData - State data array with time and value
 * @param {Array} props.actionData - Action data array with time and value
 * @param {number} props.currentTime - Current playback time
 * @param {number} props.duration - Total duration
 * @param {boolean} props.isExpanded - Whether chart is expanded
 * @param {Function} props.onToggle - Toggle expand callback
 * @param {boolean} props.hasAction - Whether action data exists
 * @param {Function} props.onSeek - Seek callback when clicking on chart
 */
function JointChart({
    name,
    stateData = [],
    actionData = [],
    currentTime = 0,
    duration = 0,
    isExpanded = false,
    onToggle,
    hasAction = false,
    onSeek,
}) {
    const chartRef = useRef(null);
    const [isChartVisible, setIsChartVisible] = useState(false);

    useEffect(() => {
        if (!isExpanded) {
            setIsChartVisible(false);
            return undefined;
        }

        const element = chartRef.current;
        if (!element || typeof IntersectionObserver === 'undefined') {
            setIsChartVisible(true);
            return undefined;
        }

        const observer = new IntersectionObserver(
            (entries) => {
                setIsChartVisible(entries.some((entry) => entry.isIntersecting));
            },
            { root: null, rootMargin: '240px 0px' }
        );
        observer.observe(element);
        return () => observer.disconnect();
    }, [isExpanded]);

    // Get current state value at currentTime
    const currentStateValue = useMemo(() => {
        if (!stateData.length) return null;
        return valueForSeriesAtTime(stateData, currentTime, `state_${name}`);
    }, [stateData, name, currentTime]);

    // Get current action value at currentTime
    const currentActionValue = useMemo(() => {
        if (!actionData.length) return null;
        return valueForSeriesAtTime(actionData, currentTime, `action_${name}`);
    }, [actionData, name, currentTime]);

    const shouldRenderChart = isExpanded && isChartVisible;

    // Merge state and action data for the chart
    const mergedData = useMemo(() => {
        if (!shouldRenderChart) return [];
        return buildJointChartData(stateData, actionData, name, hasAction);
    }, [stateData, actionData, name, hasAction, shouldRenderChart]);

    // Calculate X-axis domain to include full duration
    const xDomain = useMemo(() => {
        const maxDataTime = mergedData.length > 0
            ? Math.max(...mergedData.map(d => d.time))
            : 0;
        return [0, Math.max(maxDataTime, duration || 0)];
    }, [mergedData, duration]);

    return (
        <div ref={chartRef} className="border border-gray-200 rounded-lg overflow-hidden bg-white">
            {/* Header - always visible */}
            <button
                onClick={onToggle}
                className="w-full flex items-center justify-between px-3 py-2 hover:bg-gray-50 transition-colors"
            >
                <div className="flex items-center gap-2">
                    <span className="font-medium text-sm text-gray-700">{name}</span>
                </div>
                <div className="flex items-center gap-3">
                    {/* State value */}
                    <div className="flex items-center gap-1">
                        <div className="w-2 h-2 rounded-full" style={{ backgroundColor: STATE_COLOR }} />
                        <span className="text-xs text-gray-500 font-mono">
                            {currentStateValue !== null && currentStateValue !== undefined
                                ? currentStateValue.toFixed(4)
                                : '-'}
                        </span>
                    </div>
                    {/* Action value */}
                    {hasAction && (
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: ACTION_COLOR }} />
                            <span className="text-xs text-gray-500 font-mono">
                                {currentActionValue !== null && currentActionValue !== undefined
                                    ? currentActionValue.toFixed(4)
                                    : '-'}
                            </span>
                        </div>
                    )}
                    {isExpanded ? (
                        <MdExpandLess className="text-gray-400" size={20} />
                    ) : (
                        <MdExpandMore className="text-gray-400" size={20} />
                    )}
                </div>
            </button>

            {/* Chart - collapsible */}
            {isExpanded && (
                <div className="h-36 px-2 pb-2">
                    {shouldRenderChart ? (
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                            data={mergedData}
                            margin={{ top: 5, right: 10, left: 0, bottom: 5 }}
                            onClick={(e) => {
                                if (e && e.activeLabel !== undefined && onSeek) {
                                    onSeek(e.activeLabel);
                                }
                            }}
                            style={{ cursor: 'crosshair' }}
                        >
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis
                                dataKey="time"
                                domain={xDomain}
                                type="number"
                                tickFormatter={(value) => `${value.toFixed(0)}s`}
                                stroke="#9ca3af"
                                fontSize={10}
                                tick={{ fill: '#9ca3af' }}
                                allowDataOverflow={true}
                            />
                            <YAxis
                                stroke="#9ca3af"
                                fontSize={10}
                                tick={{ fill: '#9ca3af' }}
                                width={45}
                                tickFormatter={(value) => value.toFixed(2)}
                            />
                            <Tooltip
                                content={(props) => (
                                    <JointTooltip
                                        {...props}
                                        stateData={stateData}
                                        actionData={actionData}
                                        name={name}
                                        hasAction={hasAction}
                                    />
                                )}
                            />
                            {/* State line - Red */}
                            <Line
                                type="linear"
                                dataKey={`state_${name}`}
                                stroke={STATE_COLOR}
                                strokeWidth={1.5}
                                dot={false}
                                isAnimationActive={false}
                                name="State"
                                connectNulls
                            />
                            {/* Action line - Blue */}
                            {hasAction && (
                                <Line
                                    type="linear"
                                    dataKey={`action_${name}`}
                                    stroke={ACTION_COLOR}
                                    strokeWidth={1.5}
                                    dot={false}
                                    isAnimationActive={false}
                                    name="Action"
                                    connectNulls
                                />
                            )}
                            {/* Current time indicator line */}
                            <ReferenceLine
                                x={currentTime}
                                stroke="#22c55e"
                                strokeWidth={2}
                                strokeDasharray="none"
                            />
                            {/* Current position marker for State */}
                            {currentStateValue !== null && (
                                <ReferenceDot
                                    x={currentTime}
                                    y={currentStateValue}
                                    r={6}
                                    fill={STATE_COLOR}
                                    stroke="#fff"
                                    strokeWidth={2}
                                    isAnimationActive={false}
                                    ifOverflow="extendDomain"
                                />
                            )}
                            {/* Current position marker for Action */}
                            {hasAction && currentActionValue !== null && (
                                <ReferenceDot
                                    x={currentTime}
                                    y={currentActionValue}
                                    r={6}
                                    fill={ACTION_COLOR}
                                    stroke="#fff"
                                    strokeWidth={2}
                                    isAnimationActive={false}
                                    ifOverflow="extendDomain"
                                />
                            )}
                        </LineChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex h-full items-center justify-center rounded bg-gray-50 text-[11px] text-gray-400">
                        Chart paused off-screen
                      </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default JointChart;
