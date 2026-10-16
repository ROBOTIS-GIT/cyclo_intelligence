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

// Highlight only action language belonging to the structured decision. This is
// presentation only: it never derives or dispatches actions from generated text.
const ACTIONS = {
  wait: [String.raw`wait`, String.raw`stand by`, String.raw`keep watching`, String.raw`remain here`, String.raw`(?:keep|continue) observing`, String.raw`do not start (?:a |any |the )?mission`],
  continue: [String.raw`continue(?: (?:moving|navigation|the mission|the task))?`, String.raw`keep (?:moving|going)`, String.raw`proceed`, String.raw`resume(?: (?:navigation|the mission|the task))?`],
  pause: [String.raw`pause(?: (?:the mission|the task|navigation))?`, String.raw`stop(?: (?:moving|movement|navigation))?`, String.raw`hold (?:position|still)`],
  cancel: [String.raw`(?:cancel|abort|abandon)(?: (?:the |this )?(?:mission|task))?`, String.raw`return to (?:the )?(?:observation (?:point|spot)|patrol)`, String.raw`resume patrol`],
  next_observation: [String.raw`(?:move|go|proceed|head)(?: on)? to (?:the )?next observation(?: (?:point|waypoint|spot))?`, String.raw`next[ _]observation(?: (?:point|waypoint|spot))?`],
  unknown: [String.raw`unknown`, String.raw`uncertain`, String.raw`unclear`, String.raw`cannot (?:determine|assess|decide)`],
};
const escapeRegex = text => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

function missionPatterns(result) {
  const name = result?.selected_mission?.mission_name;
  if (!name?.trim()) return [];
  const words = name.replace(/([a-z0-9])([A-Z])/g, "$1 $2").replace(/[_-]+/g, " ").trim().split(/\s+/);
  const join = parts => parts.map(escapeRegex).join(String.raw`[ _-]+(?:the )?`);
  const patterns = [escapeRegex(name), join(words)];
  // "TurnOffTV" also appears naturally as "turn the TV off".
  if (/^turn$/i.test(words[0]) && /^(on|off)$/i.test(words[1])) {
    patterns.push(join([words[0], ...words.slice(2), words[1]]));
  }
  const key = words.join("").toLowerCase();
  const aliases = {
    turnofftv: [String.raw`turn (?:it|the TV|the television) off`, String.raw`turn off (?:the )?(?:TV|television)`],
    turnontv: [String.raw`turn (?:it|the TV|the television) on`, String.raw`turn on (?:the )?(?:TV|television)`],
    cleantable: [String.raw`(?:clean|clear)(?: up)? (?:the )?(?:dining )?table`],
    cleandish: [String.raw`(?:clean|clear)(?: up)? (?:the )?(?:dining )?table`, String.raw`(?:clean|collect) (?:the )?dishes`],
    servedish: [String.raw`(?:serve|bring|place) (?:the )?(?:dishes|plates|food|meal)`],
    dolaundry: [String.raw`(?:do|collect|wash) (?:the )?(?:laundry|clothes)`, String.raw`start (?:the )?(?:washing machine|wash cycle)`],
    trashdisposal: [String.raw`(?:take out|dispose of|remove) (?:the )?(?:trash|garbage)`],
  };
  return [...patterns, ...(aliases[key] || [])];
}

export function actionRanges(text, result) {
  const decision = result?.decision?.decision;
  const patterns = decision === "start"
    ? [...missionPatterns(result), String.raw`(?:start|execute|run|begin) (?:the |this |a )?mission`]
    : ACTIONS[decision] || [];
  if (!text || !patterns.length) return [];
  const matches = [];
  const regex = new RegExp(String.raw`\b(?:${patterns.join("|")})(?!\w)`, "gi");
  for (const match of text.matchAll(regex)) {
    const before = text.slice(Math.max(0, match.index - 60), match.index);
    // Avoid painting "do not clean the table" as an affirmative clean action.
    if (/(?:\bnot|\bnever|\bno need to|\bdon['’]t|\bcan['’]t|\bcannot|\bshouldn['’]t|\bmustn['’]t)(?:\s+\w+){0,3}\s*$/i.test(before)) continue;
    matches.push({ start: match.index, end: match.index + match[0].length });
  }
  return matches;
}

export default function ReasoningText({ text, result }) {
  const ranges = actionRanges(text, result);
  if (!ranges.length) return text;
  const content = [];
  let cursor = 0;
  for (const { start, end } of ranges) {
    content.push(text.slice(cursor, start));
    content.push(<strong className="operate-thought-action" key={start}>{text.slice(start, end)}</strong>);
    cursor = end;
  }
  content.push(text.slice(cursor));
  return <>{content}</>;
}
