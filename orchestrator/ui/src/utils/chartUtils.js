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

/**
 * LTTB (Largest Triangle Three Buckets) downsampling algorithm.
 * Preserves visual shape while reducing data points for efficient rendering.
 *
 * @param {Array} data - Array of data points
 * @param {number} threshold - Target number of points after downsampling
 * @param {string} xKey - Key for x-axis value in data objects
 * @param {string} yKey - Key for y-axis value in data objects
 * @returns {Array} Downsampled data array
 */
export function lttbDownsample(data, threshold, xKey, yKey) {
    if (threshold >= data.length || threshold === 0) {
        return data;
    }

    const sampled = [];
    const dataLength = data.length;

    // Always include first point
    sampled.push(data[0]);

    // Bucket size
    const bucketSize = (dataLength - 2) / (threshold - 2);

    let a = 0; // Previously selected point index

    for (let i = 0; i < threshold - 2; i++) {
        // Calculate bucket boundaries
        const bucketStart = Math.floor((i + 1) * bucketSize) + 1;
        const bucketEnd = Math.min(Math.floor((i + 2) * bucketSize) + 1, dataLength);

        // Calculate average point in next bucket
        const avgRangeStart = Math.floor((i + 2) * bucketSize) + 1;
        const avgRangeEnd = Math.min(Math.floor((i + 3) * bucketSize) + 1, dataLength);

        let avgX = 0;
        let avgY = 0;
        let avgCount = 0;

        for (let j = avgRangeStart; j < avgRangeEnd; j++) {
            if (j < dataLength) {
                avgX += data[j][xKey];
                avgY += data[j][yKey] || 0;
                avgCount++;
            }
        }

        if (avgCount > 0) {
            avgX /= avgCount;
            avgY /= avgCount;
        }

        // Find point in current bucket that creates largest triangle
        let maxArea = -1;
        let maxAreaIndex = bucketStart;

        const pointA = data[a];

        for (let j = bucketStart; j < bucketEnd && j < dataLength; j++) {
            // Calculate triangle area using cross product
            const area = Math.abs(
                (pointA[xKey] - avgX) * ((data[j][yKey] || 0) - pointA[yKey]) -
                (pointA[xKey] - data[j][xKey]) * (avgY - pointA[yKey])
            );

            if (area > maxArea) {
                maxArea = area;
                maxAreaIndex = j;
            }
        }

        sampled.push(data[maxAreaIndex]);
        a = maxAreaIndex;
    }

    // Always include last point
    sampled.push(data[dataLength - 1]);

    return sampled;
}

export function downsampleMultiSeriesExtrema(data, threshold, xKey, yKeys) {
    if (threshold >= data.length || threshold === 0) {
        return data;
    }

    const keys = (yKeys || []).filter(Boolean);
    if (keys.length === 0) {
        if (threshold <= 1) return data.slice(0, 1);
        return [
            ...data.slice(0, threshold - 1),
            data[data.length - 1],
        ];
    }
    if (keys.length <= 1) {
        return lttbDownsample(data, threshold, xKey, keys[0]);
    }

    const bucketCount = Math.floor((threshold - 2) / (keys.length * 2));
    if (bucketCount < 1) {
        return lttbDownsample(data, threshold, xKey, keys[0]);
    }

    const selectedIndexes = new Set([0, data.length - 1]);
    const bucketSize = (data.length - 2) / bucketCount;

    for (let bucket = 0; bucket < bucketCount; bucket++) {
        const start = Math.floor(bucket * bucketSize) + 1;
        const end = Math.min(Math.floor((bucket + 1) * bucketSize) + 1, data.length - 1);
        if (start >= end) continue;

        keys.forEach((key) => {
            let minIndex = start;
            let maxIndex = start;
            let minValue = Number(data[start]?.[key]);
            let maxValue = minValue;
            if (!Number.isFinite(minValue)) {
                minValue = 0;
                maxValue = 0;
            }

            for (let i = start + 1; i < end; i++) {
                let value = Number(data[i]?.[key]);
                if (!Number.isFinite(value)) value = 0;
                if (value < minValue) {
                    minValue = value;
                    minIndex = i;
                }
                if (value > maxValue) {
                    maxValue = value;
                    maxIndex = i;
                }
            }

            selectedIndexes.add(minIndex);
            selectedIndexes.add(maxIndex);
        });
    }

    return Array.from(selectedIndexes)
        .sort((a, b) => a - b)
        .map((index) => data[index]);
}

/**
 * Prepare chart data from timestamps, names, and values arrays.
 * Applies LTTB downsampling for large datasets.
 *
 * @param {Array<number>} timestamps - Array of timestamp values
 * @param {Array<string>} names - Array of data field names (e.g., joint names)
 * @param {Array<number>} values - Flattened array of values (length = timestamps.length * names.length)
 * @param {string} prefix - Prefix for data keys (e.g., 'state_' or 'action_')
 * @param {number} targetPoints - Target number of points after downsampling (default: 1000)
 * @returns {Array} Chart-ready data array with time and prefixed value keys
 */
export function prepareChartData(timestamps, names, values, prefix, targetPoints = 1000) {
    if (!timestamps.length || !names.length || !values.length) {
        return [];
    }

    const numFields = names.length;

    // Create full data array
    const fullData = timestamps.map((time, i) => {
        const point = { time };
        const startIdx = i * numFields;
        names.forEach((name, j) => {
            point[`${prefix}${name}`] = values[startIdx + j] || 0;
        });
        return point;
    });

    // If data is small enough, return as is
    if (fullData.length <= targetPoints) {
        return fullData;
    }

    const keys = names.map((name) => `${prefix}${name}`);
    return downsampleMultiSeriesExtrema(fullData, targetPoints, 'time', keys);
}

/**
 * Format file size to human readable format.
 *
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted file size string (e.g., "1.5 MB")
 */
export function formatFileSize(bytes) {
    if (bytes === 0 || !bytes) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(1)} ${units[i]}`;
}

const pad2 = (value) => String(value).padStart(2, '0');

const normalizeIsoDateTimeAsUtc = (value) => {
    const text = String(value).trim();
    if (/^\d{4}-\d{2}-\d{2}$/.test(text)) {
        return `${text}T00:00:00Z`;
    }
    if (/^\d{4}-\d{2}-\d{2}T/.test(text) && !/(Z|[+-]\d{2}:?\d{2})$/i.test(text)) {
        return `${text}Z`;
    }
    return text;
};

/**
 * Format datetime to a locale-independent UTC timestamp.
 *
 * @param {string} isoString - ISO format datetime string
 * @returns {string} Formatted datetime string
 */
export function formatDateTime(isoString) {
    if (!isoString) return '-';
    try {
        const date = new Date(normalizeIsoDateTimeAsUtc(isoString));
        if (Number.isNaN(date.getTime())) return isoString;
        return [
            `${date.getUTCFullYear()}-${pad2(date.getUTCMonth() + 1)}-${pad2(date.getUTCDate())}`,
            `${pad2(date.getUTCHours())}:${pad2(date.getUTCMinutes())}:${pad2(date.getUTCSeconds())}`,
            'UTC',
        ].join(' ');
    } catch {
        return isoString;
    }
}

/**
 * Format time in seconds to MM:SS format.
 *
 * @param {number} seconds - Time in seconds
 * @returns {string} Formatted time string (e.g., "01:30")
 */
export function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}
