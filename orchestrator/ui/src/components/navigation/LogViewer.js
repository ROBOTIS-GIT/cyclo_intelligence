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
// Author: Howon Kim

"use client";
import { useRef, useState, useMemo, useLayoutEffect, useCallback } from "react";
import Convert from "ansi-to-html";
import { useTheme } from "../../contexts/ThemeContext";
export default function LogViewer({ lines, autoScroll = true, className = "", }) {
    const scrollRef = useRef(null);
    const isAtBottomRef = useRef(true);
    const [showScrollButton, setShowScrollButton] = useState(false);
    const { theme } = useTheme();
    const convert = useMemo(() => {
        const isDark = theme === "dark";
        return new Convert({
            fg: isDark ? "#d4d4d4" : "#333333",
            bg: isDark ? "#1e1e1e" : "#ffffff",
            newline: false,
            escapeXML: true,
            stream: false,
            colors: isDark
                ? {
                    0: "#000000", 1: "#cd3131", 2: "#0dbc79", 3: "#e5e510",
                    4: "#2472c8", 5: "#bc3fbc", 6: "#11a8cd", 7: "#e5e5e5",
                    8: "#666666", 9: "#f14c4c", 10: "#23d18b", 11: "#f5f543",
                    12: "#3b8eea", 13: "#d670d6", 14: "#29b8db", 15: "#e5e5e5",
                }
                : {
                    0: "#000000", 1: "#cd3131", 2: "#0dbc79", 3: "#e5e510",
                    4: "#2472c8", 5: "#bc3fbc", 6: "#11a8cd", 7: "#333333",
                    8: "#666666", 9: "#f14c4c", 10: "#23d18b", 11: "#f5f543",
                    12: "#3b8eea", 13: "#d670d6", 14: "#29b8db", 15: "#333333",
                },
        });
    }, [theme]);
    const htmlLines = useMemo(() => lines.map((line) => convert.toHtml(line)), [lines, convert]);
    useLayoutEffect(() => {
        const container = scrollRef.current;
        if (!container)
            return;
        if (autoScroll && isAtBottomRef.current) {
            container.scrollTop = container.scrollHeight;
        }
    }, [htmlLines, autoScroll]);
    const handleScroll = useCallback(() => {
        const container = scrollRef.current;
        if (!container)
            return;
        const { scrollTop, scrollHeight, clientHeight } = container;
        const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
        isAtBottomRef.current = distanceFromBottom < 50;
        setShowScrollButton(!isAtBottomRef.current);
    }, []);
    return (<div className={`relative flex flex-col ${className}`} style={{ height: "100%", minHeight: 0, position: "relative", overflow: "hidden" }}>
      <div ref={scrollRef} onScroll={handleScroll} className="p-4 rounded" style={{
            fontFamily: "monospace",
            fontSize: "12px",
            backgroundColor: theme === "dark" ? "#1e1e1e" : "#ffffff",
            color: theme === "dark" ? "#d4d4d4" : "#333333",
            border: "1px solid var(--vscode-panel-border)",
            overflowY: "auto",
            overflowX: "auto",
            flex: 1,
            height: 0,
            minHeight: 0,
            maxHeight: "100%",
            position: "relative",
            overflowAnchor: "none",
            scrollBehavior: "auto",
        }}>
        {htmlLines.length === 0 ? (<div style={{ color: theme === "dark" ? "#666666" : "#999999" }}>No logs available</div>) : (htmlLines.map((html, i) => (<div key={i} style={{ whiteSpace: "pre-wrap", wordBreak: "break-all", minHeight: "1em", fontFamily: "monospace" }} dangerouslySetInnerHTML={{ __html: html || "\u00a0" }}/>)))}
      </div>

      {showScrollButton && autoScroll && (<button onClick={() => {
                if (scrollRef.current) {
                    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
                    isAtBottomRef.current = true;
                    setShowScrollButton(false);
                }
            }} className="absolute px-3 py-1 rounded text-xs shadow-md opacity-90 hover:opacity-100" style={{
                bottom: "16px",
                right: "16px",
                backgroundColor: "var(--vscode-button-background, #007acc)",
                color: "var(--vscode-button-foreground, white)",
                border: "none",
                cursor: "pointer",
                transition: "all 0.2s",
                zIndex: 100,
                pointerEvents: "auto",
            }}>
          ⬇ Scroll to Bottom
        </button>)}
    </div>);
}

