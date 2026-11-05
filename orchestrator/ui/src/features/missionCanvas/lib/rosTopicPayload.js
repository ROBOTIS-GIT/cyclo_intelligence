// Copyright 2026 ROBOTIS CO., LTD.
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
// Author: Seongwoo Kim

export function topicPayload(value) {
  if (!value || typeof value !== "object") return null;
  if (value.available === false) return null;
  return "data" in value ? value.data : value;
}

export function messageData(value) {
  const data = topicPayload(value);
  return data && typeof data === "object" ? data : null;
}

export function hasTopicMessage(value) {
  if (!value || typeof value !== "object") return false;
  if (value.available === false) return false;
  if (value.available === true) return true;
  return topicPayload(value) !== null;
}

export function rosStringData(value) {
  const payload = topicPayload(value);
  if (typeof payload === "string") return payload;
  if (payload && typeof payload.data === "string") return payload.data;
  return "";
}
