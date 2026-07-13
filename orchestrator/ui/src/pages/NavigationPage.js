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
// Author: Howon Kim

"use client";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { cancelNavigateToPoseGoal, getServiceStatus, saveNavigationMap, sendNavigateToPoseGoal, startNavigation, stopNavigation, } from "../utils/navigationApi";
import { useNavigationRosPublisher, useNavigationRosTopic } from "../hooks/useNavigationRosTopic";
import { MapEditorControls, useMapEditor } from "../components/navigation/MapEditor";
import { MapViewer } from "../components/navigation/MapViewer";
import { NavigationSidePanel, } from "../components/navigation/NavigationSidePanel";
import { NavigationToolbar } from "../components/navigation/NavigationToolbar";
import { mergeTfMessages, orientationFromYaw, poseFromBaseLinkTf, tfMessageFromBuffer, updateTfBuffer, yawFromPose, } from "../utils/navigationTf";
import FixedLogPanel from "../components/navigation/FixedLogPanel";
const NAVIGATION_SERVICE = "ai_worker_navigation";
const SERVICE_MODE_STORAGE_KEY = "cyclo.navigation.serviceMode";
const STATUS_POLL_MS = 10000;
const DEFAULT_MAP_NAME = "map";
const LOG_PANEL_DEFAULT_WIDTH = 420;
const LOG_PANEL_MIN_WIDTH = 320;
const LOG_PANEL_MAX_WIDTH = 800;
const MAP_PANEL_DEFAULT_WIDTH = 960;
const MAP_PANEL_MIN_WIDTH = 420;
const MAP_PANEL_MAX_WIDTH = 1200;
const LAYERS_PANEL_WIDTH = 200;
const CONTENT_GRID_GAP_PX = 16;
const MAP_RESIZE_HANDLE_WIDTH_PX = 8;
const LOG_RESIZE_HANDLE_WIDTH_PX = 8;
const ROS2_WS_FAST_TOPIC_OPTIONS = { throttleMs: 100 };
const ROS2_WS_LOCAL_COSTMAP_OPTIONS = { throttleMs: 200 };
const GOAL_REACHED_XY_TOLERANCE_M = 0.2;
const GOAL_REACHED_YAW_TOLERANCE_RAD = 0.4;
const IDLE_LAYER_PRESET = {
    map: false,
    globalCostmap: false,
    localCostmap: false,
    scan: false,
    globalPlan: false,
    goalPose: false,
    tf: false,
    robotModel: false,
};
const MAPPING_LAYER_PRESET = {
    map: true,
    globalCostmap: false,
    localCostmap: false,
    scan: true,
    globalPlan: false,
    goalPose: false,
    tf: false,
    robotModel: true,
};
const NAVIGATION_LAYER_PRESET = {
    map: true,
    globalCostmap: true,
    localCostmap: true,
    scan: true,
    globalPlan: true,
    goalPose: true,
    tf: false,
    robotModel: true,
};
const DISPLAY_TOPICS = [
    "/map",
    "/global_costmap/costmap",
    "/local_costmap/costmap",
    "/local_costmap/published_footprint",
    "/scan",
    "/plan",
    "/goal_pose",
    "/tf",
];
function messageData(value) {
    if (!value || typeof value !== "object")
        return null;
    const outer = value;
    if (outer.available === false)
        return null;
    const data = "data" in outer ? outer.data : outer;
    if (!data || typeof data !== "object")
        return null;
    return data;
}
function normalizeAngle(angle) {
    return Math.atan2(Math.sin(angle), Math.cos(angle));
}
function isGoalReached(current, goal) {
    var _a, _b, _c, _d;
    const currentPosition = current === null || current === void 0 ? void 0 : current.position;
    const goalPose = goal === null || goal === void 0 ? void 0 : goal.pose;
    const goalPosition = goalPose === null || goalPose === void 0 ? void 0 : goalPose.position;
    if (!currentPosition || !goalPosition)
        return false;
    const dx = Number((_a = currentPosition.x) !== null && _a !== void 0 ? _a : 0) - Number((_b = goalPosition.x) !== null && _b !== void 0 ? _b : 0);
    const dy = Number((_c = currentPosition.y) !== null && _c !== void 0 ? _c : 0) - Number((_d = goalPosition.y) !== null && _d !== void 0 ? _d : 0);
    const distance = Math.hypot(dx, dy);
    const yawError = Math.abs(normalizeAngle(yawFromPose(current) - yawFromPose(goalPose)));
    return distance <= GOAL_REACHED_XY_TOLERANCE_M && yawError <= GOAL_REACHED_YAW_TOLERANCE_RAD;
}
function rosTimestampNow() {
    const nowMs = Date.now();
    const sec = Math.floor(nowMs / 1000);
    const nanosec = Math.floor((nowMs % 1000) * 1000000);
    return { sec, nanosec };
}
function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}
function readStoredServiceMode() {
    if (typeof window === "undefined")
        return "navigation";
    return window.localStorage.getItem(SERVICE_MODE_STORAGE_KEY) === "mapping" ? "mapping" : "navigation";
}
export default function NavigationPage() {
    var _a, _b, _c, _d;
    const posePublishBusyRef = useRef(false);
    const logResizeRef = useRef(null);
    const mapResizeRef = useRef(null);
    const contentGridRef = useRef(null);
    const statusLoadingRef = useRef(false);
    const tfBufferRef = useRef(new Map());
    const [status, setStatus] = useState(null);
    const [mapName, setMapName] = useState(DEFAULT_MAP_NAME);
    const [busy, setBusy] = useState(null);
    const [message, setMessage] = useState("Ready");
    const [lastGoalPose, setLastGoalPose] = useState(null);
    const [hideReachedGoalPose, setHideReachedGoalPose] = useState(false);
    const [lastBaseLinkPose, setLastBaseLinkPose] = useState(null);
    const [showMap, setShowMap] = useState(false);
    const [showGlobalCostmap, setShowGlobalCostmap] = useState(false);
    const [showLocalCostmap, setShowLocalCostmap] = useState(false);
    const [showScan, setShowScan] = useState(false);
    const [showGlobalPlan, setShowGlobalPlan] = useState(false);
    const [showGoalPose, setShowGoalPose] = useState(false);
    const [showTf, setShowTf] = useState(false);
    const [showRobotModel, setShowRobotModel] = useState(false);
    const [showLogs, setShowLogs] = useState(false);
    const [logPanelWidth, setLogPanelWidth] = useState(LOG_PANEL_DEFAULT_WIDTH);
    const [mapPanelWidth, setMapPanelWidth] = useState(MAP_PANEL_DEFAULT_WIDTH);
    const [clickMode, setClickMode] = useState("view");
    const [posePublishBusy, setPosePublishBusy] = useState(false);
    const [showPgmFix, setShowPgmFix] = useState(false);
    const [tfBufferRevision, setTfBufferRevision] = useState(0);
    const [serviceMode, setServiceMode] = useState(readStoredServiceMode);
    const publishRosTopic = useNavigationRosPublisher();
    const mapEditor = useMapEditor({
        open: showPgmFix,
        mapName,
        onMessage: setMessage,
    });
    const running = (_a = status === null || status === void 0 ? void 0 : status.is_up) !== null && _a !== void 0 ? _a : false;
    const navigationTopicsActive = running && busy !== "Stop" && !showPgmFix;
    const { topicData: mapData } = useNavigationRosTopic(navigationTopicsActive && (showMap || clickMode !== "view") ? "/map" : null);
    const { topicData: globalCostmapData } = useNavigationRosTopic(navigationTopicsActive && showGlobalCostmap ? "/global_costmap/costmap" : null);
    const { topicData: localCostmapData } = useNavigationRosTopic(navigationTopicsActive && showLocalCostmap ? "/local_costmap/costmap" : null, ROS2_WS_LOCAL_COSTMAP_OPTIONS);
    const { topicData: footprintData } = useNavigationRosTopic(navigationTopicsActive && showRobotModel ? "/local_costmap/published_footprint" : null, ROS2_WS_FAST_TOPIC_OPTIONS);
    const { topicData: scanData } = useNavigationRosTopic(navigationTopicsActive && showScan ? "/scan" : null, ROS2_WS_FAST_TOPIC_OPTIONS);
    const { topicData: amclData } = useNavigationRosTopic(navigationTopicsActive ? "/amcl_pose" : null, ROS2_WS_FAST_TOPIC_OPTIONS);
    const { topicData: planData } = useNavigationRosTopic(navigationTopicsActive && showGlobalPlan ? "/plan" : null);
    const { topicData: goalPoseData } = useNavigationRosTopic(navigationTopicsActive && showGoalPose ? "/goal_pose" : null);
    const { topicData: tfData } = useNavigationRosTopic(navigationTopicsActive ? "/tf" : null, ROS2_WS_FAST_TOPIC_OPTIONS);
    const { topicData: tfStaticData } = useNavigationRosTopic(navigationTopicsActive ? "/tf_static" : null);
    const map = useMemo(() => messageData(mapData), [mapData]);
    const globalCostmap = useMemo(() => messageData(globalCostmapData), [globalCostmapData]);
    const localCostmap = useMemo(() => messageData(localCostmapData), [localCostmapData]);
    const footprint = useMemo(() => messageData(footprintData), [footprintData]);
    const scan = useMemo(() => messageData(scanData), [scanData]);
    const amclPose = useMemo(() => messageData(amclData), [amclData]);
    const plan = useMemo(() => messageData(planData), [planData]);
    const topicGoalPose = useMemo(() => messageData(goalPoseData), [goalPoseData]);
    const tf = useMemo(() => messageData(tfData), [tfData]);
    const tfStatic = useMemo(() => messageData(tfStaticData), [tfStaticData]);
    const latestTf = useMemo(() => mergeTfMessages(tfStatic, tf), [tf, tfStatic]);
    // tfBufferRevision intentionally causes a render when mutable TF data changes.
    void tfBufferRevision;
    const bufferedTf = (_b = tfMessageFromBuffer(tfBufferRef.current)) !== null && _b !== void 0 ? _b : latestTf;
    const fallbackPose = (_d = (_c = amclPose === null || amclPose === void 0 ? void 0 : amclPose.pose) === null || _c === void 0 ? void 0 : _c.pose) !== null && _d !== void 0 ? _d : null;
    const goalPose = hideReachedGoalPose ? null : (lastGoalPose !== null && lastGoalPose !== void 0 ? lastGoalPose : topicGoalPose);
    const topicBaseLinkPose = useMemo(() => poseFromBaseLinkTf(bufferedTf), [bufferedTf]);
    const baseLinkPose = topicBaseLinkPose !== null && topicBaseLinkPose !== void 0 ? topicBaseLinkPose : lastBaseLinkPose;
    const currentPose = baseLinkPose !== null && baseLinkPose !== void 0 ? baseLinkPose : fallbackPose;
    const mode = running ? "running" : "idle";
    const trimmedMapName = mapName.trim();
    const hasMapName = trimmedMapName.length > 0;
    const displayedMap = showPgmFix ? mapEditor.map : map;
    const mapViewKey = showPgmFix
        ? `editor:${mapEditor.selectedPath || "none"}`
        : `ros:${mode}:${mapName || DEFAULT_MAP_NAME}`;
    const layersPanelWidth = LAYERS_PANEL_WIDTH;
    const getMaxMapPanelWidth = useCallback(() => {
        var _a, _b;
        const gridWidth = (_b = (_a = contentGridRef.current) === null || _a === void 0 ? void 0 : _a.clientWidth) !== null && _b !== void 0 ? _b : 0;
        if (!gridWidth)
            return MAP_PANEL_MAX_WIDTH;
        const reservedWidth = showLogs
            ? layersPanelWidth + logPanelWidth + MAP_RESIZE_HANDLE_WIDTH_PX + LOG_RESIZE_HANDLE_WIDTH_PX + CONTENT_GRID_GAP_PX * 4
            : 300 + MAP_RESIZE_HANDLE_WIDTH_PX + CONTENT_GRID_GAP_PX * 2;
        return clamp(gridWidth - reservedWidth, MAP_PANEL_MIN_WIDTH, MAP_PANEL_MAX_WIDTH);
    }, [layersPanelWidth, logPanelWidth, showLogs]);
    const contentGridStyle = {
        "--map-panel-width": `${mapPanelWidth}px`,
        ...(showLogs
            ? {
                "--layers-panel-width": `${layersPanelWidth}px`,
                "--log-panel-min-width": `${logPanelWidth}px`,
            }
            : {}),
    };
    const layerToggles = useMemo(() => [
        { id: "map", label: "Map", checked: showMap, onChange: setShowMap },
        {
            id: "globalCostmap",
            label: "Global costmap",
            checked: showGlobalCostmap,
            onChange: setShowGlobalCostmap,
        },
        {
            id: "localCostmap",
            label: "Local costmap",
            checked: showLocalCostmap,
            onChange: setShowLocalCostmap,
        },
        { id: "scan", label: "Lidar", checked: showScan, onChange: setShowScan },
        {
            id: "globalPlan",
            label: "Global plan",
            checked: showGlobalPlan,
            onChange: setShowGlobalPlan,
        },
        { id: "goalPose", label: "Goal Pose", checked: showGoalPose, onChange: setShowGoalPose },
        { id: "tf", label: "TF", checked: showTf, onChange: setShowTf },
        {
            id: "robotModel",
            label: "Robot Model",
            checked: showRobotModel,
            onChange: setShowRobotModel,
        },
    ], [
        showGlobalCostmap,
        showGlobalPlan,
        showGoalPose,
        showLocalCostmap,
        showMap,
        showRobotModel,
        showScan,
        showTf,
    ]);
    const topicRows = useMemo(() => DISPLAY_TOPICS.map((topic) => {
        var _a, _b, _c;
        if (topic === "/map")
            return { topic, isLive: !!map };
        if (topic === "/global_costmap/costmap")
            return { topic, isLive: !!globalCostmap };
        if (topic === "/local_costmap/costmap")
            return { topic, isLive: !!localCostmap };
        if (topic === "/local_costmap/published_footprint") {
            return { topic, isLive: !!((_b = (_a = footprint === null || footprint === void 0 ? void 0 : footprint.polygon) === null || _a === void 0 ? void 0 : _a.points) === null || _b === void 0 ? void 0 : _b.length) };
        }
        if (topic === "/scan")
            return { topic, isLive: !!scan };
        if (topic === "/plan")
            return { topic, isLive: !!plan };
        if (topic === "/goal_pose")
            return { topic, isLive: !!goalPose };
        if (topic === "/tf")
            return { topic, isLive: !!((_c = tf === null || tf === void 0 ? void 0 : tf.transforms) === null || _c === void 0 ? void 0 : _c.length) };
        return { topic, isLive: false };
    }), [
        footprint,
        globalCostmap,
        goalPose,
        localCostmap,
        map,
        plan,
        scan,
        tf,
    ]);
    const applyLayerPreset = useCallback((preset) => {
        setShowMap(preset.map);
        setShowGlobalCostmap(preset.globalCostmap);
        setShowLocalCostmap(preset.localCostmap);
        setShowScan(preset.scan);
        setShowGlobalPlan(preset.globalPlan);
        setShowGoalPose(preset.goalPose);
        setShowTf(preset.tf);
        setShowRobotModel(preset.robotModel);
    }, []);
    useEffect(() => {
        window.localStorage.setItem(SERVICE_MODE_STORAGE_KEY, serviceMode);
    }, [serviceMode]);
    useEffect(() => {
        const updatedStatic = updateTfBuffer(tfBufferRef.current, tfStatic);
        const updatedDynamic = updateTfBuffer(tfBufferRef.current, tf);
        if (updatedStatic || updatedDynamic) {
            setTfBufferRevision((value) => value + 1);
        }
    }, [tf, tfStatic]);
    useEffect(() => {
        if (topicBaseLinkPose)
            setLastBaseLinkPose(topicBaseLinkPose);
    }, [topicBaseLinkPose]);
    useEffect(() => {
        if (!lastGoalPose || !isGoalReached(currentPose, lastGoalPose))
            return;
        setLastGoalPose(null);
        setHideReachedGoalPose(true);
        setMessage("Goal reached");
    }, [currentPose, lastGoalPose]);
    useEffect(() => {
        if (running) {
            applyLayerPreset(serviceMode === "mapping" ? MAPPING_LAYER_PRESET : NAVIGATION_LAYER_PRESET);
            return;
        }
        applyLayerPreset(IDLE_LAYER_PRESET);
    }, [applyLayerPreset, running, serviceMode]);
    useEffect(() => {
        const resize = () => {
            setMapPanelWidth((width) => Math.min(width, getMaxMapPanelWidth()));
        };
        resize();
        window.addEventListener("resize", resize);
        return () => {
            window.removeEventListener("resize", resize);
        };
    }, [getMaxMapPanelWidth]);
    const loadStatus = useCallback(async () => {
        if (statusLoadingRef.current || document.visibilityState === "hidden")
            return;
        statusLoadingRef.current = true;
        try {
            const next = await getServiceStatus();
            setStatus(next);
        }
        catch (_a) {
            setStatus((current) => current);
        }
        finally {
            statusLoadingRef.current = false;
        }
    }, []);
    useEffect(() => {
        void loadStatus();
        const interval = setInterval(loadStatus, STATUS_POLL_MS);
        const handleVisibilityChange = () => {
            if (document.visibilityState === "visible")
                void loadStatus();
        };
        document.addEventListener("visibilitychange", handleVisibilityChange);
        return () => {
            clearInterval(interval);
            document.removeEventListener("visibilitychange", handleVisibilityChange);
        };
    }, [loadStatus]);
    const runCommand = useCallback(async (label, action) => {
        setBusy(label);
        try {
            const nextMessage = await action();
            setMessage(nextMessage || `${label} complete`);
        }
        catch (error) {
            setMessage(error instanceof Error ? error.message : `${label} failed`);
        }
        finally {
            setBusy(null);
            void loadStatus();
        }
    }, [loadStatus]);
    const startMapping = useCallback(async () => {
        setServiceMode("mapping");
        await startNavigation("map", mapName);
        applyLayerPreset(MAPPING_LAYER_PRESET);
    }, [applyLayerPreset, mapName]);
    const startSavedMapNavigation = useCallback(async () => {
        setServiceMode("navigation");
        await startNavigation("nav", mapName);
        applyLayerPreset(NAVIGATION_LAYER_PRESET);
    }, [applyLayerPreset, mapName]);
    const stopNavigationService = useCallback(async () => {
        await stopNavigation();
        setStatus((current) => current ? { ...current, is_up: false, pid: null } : current);
        applyLayerPreset(IDLE_LAYER_PRESET);
    }, [applyLayerPreset]);
    const saveMap = useCallback(async () => {
        if (!trimmedMapName)
            return "Map name required";
        await saveNavigationMap(trimmedMapName);
    }, [trimmedMapName]);
    const sendGoal = useCallback(async (x, y, yaw) => {
        const orientation = orientationFromYaw(yaw);
        const poseStamped = {
            header: {
                frame_id: "map",
                stamp: rosTimestampNow(),
            },
            pose: {
                position: { x, y, z: 0 },
                orientation,
            },
        };
        setLastGoalPose({
            header: { frame_id: "map" },
            pose: poseStamped.pose,
        });
        setHideReachedGoalPose(false);
        try {
            await sendNavigateToPoseGoal({ pose: poseStamped });
            setMessage(`Goal ${x.toFixed(2)}, ${y.toFixed(2)}, yaw ${(yaw * 180 / Math.PI).toFixed(0)} deg`);
        }
        catch (error) {
            setMessage(error instanceof Error ? error.message : "Goal publish failed");
        }
    }, []);
    const cancelGoal = useCallback(async () => {
        await cancelNavigateToPoseGoal();
        setLastGoalPose(null);
        setHideReachedGoalPose(true);
    }, []);
    const sendInitialPose = useCallback(async (x, y, yaw) => {
        const orientation = orientationFromYaw(yaw);
        try {
            await publishRosTopic(
              "/initialpose",
              "geometry_msgs/msg/PoseWithCovarianceStamped",
              {
                    header: {
                        frame_id: "map",
                        stamp: rosTimestampNow(),
                    },
                    pose: {
                        pose: {
                            position: { x, y, z: 0 },
                            orientation,
                        },
                        covariance: [
                            0.25, 0, 0, 0, 0, 0,
                            0, 0.25, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0.0685,
                        ],
                    },
              }
            );
            setMessage(`Initial pose ${x.toFixed(2)}, ${y.toFixed(2)}, yaw ${(yaw * 180 / Math.PI).toFixed(0)} deg`);
        }
        catch (error) {
            setMessage(error instanceof Error ? error.message : "Initial pose publish failed");
        }
    }, [publishRosTopic]);
    const handleMapPose = useCallback((x, y, yaw) => {
        if (clickMode === "view")
            return;
        if (posePublishBusyRef.current)
            return;
        posePublishBusyRef.current = true;
        setPosePublishBusy(true);
        const publish = clickMode === "initial" ? sendInitialPose(x, y, yaw) : sendGoal(x, y, yaw);
        setClickMode("view");
        void publish.finally(() => {
            posePublishBusyRef.current = false;
            setPosePublishBusy(false);
        });
    }, [clickMode, sendGoal, sendInitialPose]);
    const handleMapResizePointerDown = useCallback((event) => {
        event.preventDefault();
        mapResizeRef.current = { startX: event.clientX, startWidth: mapPanelWidth };
        const handlePointerMove = (moveEvent) => {
            const resizeStart = mapResizeRef.current;
            if (!resizeStart)
                return;
            const nextWidth = resizeStart.startWidth + moveEvent.clientX - resizeStart.startX;
            setMapPanelWidth(clamp(nextWidth, MAP_PANEL_MIN_WIDTH, getMaxMapPanelWidth()));
        };
        const handlePointerUp = () => {
            mapResizeRef.current = null;
            document.body.style.userSelect = "";
            window.removeEventListener("pointermove", handlePointerMove);
            window.removeEventListener("pointerup", handlePointerUp);
        };
        document.body.style.userSelect = "none";
        window.addEventListener("pointermove", handlePointerMove);
        window.addEventListener("pointerup", handlePointerUp, { once: true });
    }, [getMaxMapPanelWidth, mapPanelWidth]);
    const handleLogResizePointerDown = useCallback((event) => {
        event.preventDefault();
        logResizeRef.current = { startX: event.clientX, startWidth: logPanelWidth };
        const handlePointerMove = (moveEvent) => {
            const resizeStart = logResizeRef.current;
            if (!resizeStart)
                return;
            const nextWidth = resizeStart.startWidth + resizeStart.startX - moveEvent.clientX;
            setLogPanelWidth(clamp(nextWidth, LOG_PANEL_MIN_WIDTH, LOG_PANEL_MAX_WIDTH));
            setMapPanelWidth((width) => Math.min(width, getMaxMapPanelWidth()));
        };
        const handlePointerUp = () => {
            logResizeRef.current = null;
            document.body.style.userSelect = "";
            window.removeEventListener("pointermove", handlePointerMove);
            window.removeEventListener("pointerup", handlePointerUp);
        };
        document.body.style.userSelect = "none";
        window.addEventListener("pointermove", handlePointerMove);
        window.addEventListener("pointerup", handlePointerUp, { once: true });
    }, [getMaxMapPanelWidth, logPanelWidth]);
    return (<div className="navigation-page h-full min-h-[520px] flex flex-col overflow-hidden p-4">
      <header className="shrink-0 border-b pb-3 mb-4 flex flex-col xl:flex-row xl:items-center xl:justify-between gap-3" style={{ borderColor: "var(--vscode-panel-border)" }}>
        <div className="min-w-0">
          <h1 className="text-base font-semibold" style={{ color: "var(--vscode-foreground)" }}>
            Navigation
          </h1>
          <div className="mt-1 text-xs" style={{ color: "var(--vscode-descriptionForeground)" }}>
            {message}
          </div>
        </div>
        <NavigationToolbar busy={busy} clickMode={clickMode} mapName={mapName} mode={mode} running={running} hasMapName={hasMapName} showLogs={showLogs} showPgmFix={showPgmFix} onCancel={() => runCommand("Cancel", cancelGoal)} onFixToggle={() => setShowPgmFix((value) => !value)} onMapping={() => runCommand("Mapping", startMapping)} onNavigation={() => runCommand("Navigation", startSavedMapNavigation)} onSaveMap={() => runCommand("Save map", saveMap)} onStop={() => runCommand("Stop", stopNavigationService)} setClickMode={setClickMode} setMapName={setMapName} setShowLogs={setShowLogs}/>
      </header>
      {showPgmFix && (<div className="shrink-0 border mb-4 p-3" style={{
                borderColor: "var(--vscode-panel-border)",
                backgroundColor: "var(--vscode-sideBar-background)",
            }}>
          <MapEditorControls files={mapEditor.files} selectedPath={mapEditor.selectedPath} setSelectedPath={mapEditor.setSelectedPath} tool={mapEditor.tool} setTool={mapEditor.setTool} brushSize={mapEditor.brushSize} setBrushSize={mapEditor.setBrushSize} busy={mapEditor.busy} image={mapEditor.image} dirty={mapEditor.dirty} canUndo={mapEditor.canUndo} undo={mapEditor.undo} save={mapEditor.save}/>
        </div>)}
      <div ref={contentGridRef} className={[
            "flex-1 min-h-0 grid grid-cols-1 gap-4",
            showLogs
                ? "xl:grid-cols-[var(--map-panel-width)_8px_var(--layers-panel-width)_8px_minmax(var(--log-panel-min-width),1fr)]"
                : "xl:grid-cols-[var(--map-panel-width)_8px_minmax(300px,1fr)]",
        ].join(" ")} style={contentGridStyle}>
        <MapViewer map={displayedMap} globalCostmap={showPgmFix ? null : globalCostmap} localCostmap={showPgmFix ? null : localCostmap} scan={showPgmFix ? null : scan} pose={showPgmFix ? null : currentPose} plan={showPgmFix ? null : plan} goalPose={showPgmFix ? null : goalPose} footprint={showPgmFix ? null : footprint} tf={showPgmFix ? null : bufferedTf} showMap={showPgmFix ? true : showMap} showGlobalCostmap={showPgmFix ? false : showGlobalCostmap} showLocalCostmap={showPgmFix ? false : showLocalCostmap} showScan={showPgmFix ? false : showScan} showGlobalPlan={showPgmFix ? false : showGlobalPlan} showGoalPose={showPgmFix ? false : showGoalPose} showTf={showPgmFix ? false : showTf} showRobotModel={showPgmFix ? false : showRobotModel} interactionDisabled={posePublishBusy || (showPgmFix && mapEditor.busy)} interactionMode={showPgmFix ? "view" : clickMode} editorActive={showPgmFix && !!mapEditor.map && mapEditor.tool !== "view"} viewKey={mapViewKey} waitingLabel={showPgmFix ? "Select a PGM" : "Waiting for /map"} onEditorMapPoint={mapEditor.editAtMapPoint} onMapPose={handleMapPose}/>
        <div className="hidden xl:block min-h-0 cursor-col-resize" onPointerDown={handleMapResizePointerDown} title="Resize map" aria-label="Resize map" role="separator" style={{ backgroundColor: "var(--vscode-panel-border)" }}/>
        <NavigationSidePanel layerToggles={layerToggles} mapName={mapName} status={status} topicRows={topicRows}/>
        {showLogs && (<div role="separator" aria-label="Resize log panel" aria-orientation="vertical" className="hidden xl:flex min-h-0 cursor-col-resize items-stretch justify-center" onPointerDown={handleLogResizePointerDown}>
            <div className="w-px" style={{ backgroundColor: "var(--vscode-panel-border)" }}/>
          </div>)}
        {showLogs && (<div className="min-h-[320px] min-w-0">
            <FixedLogPanel service={NAVIGATION_SERVICE}/>
          </div>)}
      </div>
    </div>);
}
