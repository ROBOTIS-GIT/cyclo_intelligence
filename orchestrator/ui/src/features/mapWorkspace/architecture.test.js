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

// Feature ownership is checked from syntax, not text mentions or import depth.
import fs from "fs";
import path from "path";
import { parse } from "@babel/parser";

const featureRoot = path.resolve(__dirname, "..");
function sourceFiles(directory) {
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap(entry => {
    const file = path.join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(file);
    return /\.[cm]?[jt]sx?$/.test(file) && !/\.(test|spec)\.[jt]sx?$/.test(file) ? [file] : [];
  });
}
function importSources(source) {
  const syntax = parse(source, { sourceType: "unambiguous", plugins: ["jsx"] });
  const imports = [];
  function visit(node) {
    if (!node || typeof node !== "object") return;
    if (["ImportDeclaration", "ExportNamedDeclaration", "ExportAllDeclaration", "ImportExpression"].includes(node.type)
        && node.source?.type === "StringLiteral") imports.push(node.source.value);
    if (node.type === "CallExpression"
        && (node.callee?.type === "Import" || (node.callee?.type === "Identifier" && node.callee.name === "require"))
        && node.arguments?.[0]?.type === "StringLiteral") imports.push(node.arguments[0].value);
    Object.values(node).forEach(value => {
      if (Array.isArray(value)) value.forEach(visit);
      else if (value && typeof value === "object") visit(value);
    });
  }
  visit(syntax);
  return imports;
}

test.each([
  ["autonomyCanvas", "missionCanvas"],
  ["missionCanvas", "autonomyCanvas"],
  ["actionCanvas", "missionCanvas"],
  ["actionCanvas", "autonomyCanvas"],
  ["btEditor", "actionCanvas"],
  ["btEditor", "missionCanvas"],
  ["btEditor", "autonomyCanvas"],
])("%s does not import implementation from %s", (owner, other) => {
  const forbidden = path.join(featureRoot, other) + path.sep;
  const violations = sourceFiles(path.join(featureRoot, owner)).flatMap(file =>
    importSources(fs.readFileSync(file, "utf8")).flatMap(source => {
      const resolved = source.startsWith(".") ? path.resolve(path.dirname(file), source)
        : source.startsWith("features/") ? path.resolve(featureRoot, "..", source) : null;
      return resolved && (resolved === forbidden.slice(0, -1) || resolved.startsWith(forbidden))
        ? [`${path.relative(featureRoot, file)} -> ${source}`] : [];
    }));
  expect(violations).toEqual([]);
});

test("ownership inspection ignores comments and documentation strings", () => {
  expect(importSources(`
    // import legacy from '../missionCanvas/legacy';
    const notes = "import legacy from '../missionCanvas/legacy'";
    import shared from '../mapWorkspace/shared';
    export { tool } from './tool';
    const loader = () => import('./lazy');
  `)).toEqual(["../mapWorkspace/shared", "./tool", "./lazy"]);
});
