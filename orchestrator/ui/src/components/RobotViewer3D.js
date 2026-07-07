import React, { useRef, useEffect, useMemo, useState, useCallback, useImperativeHandle, forwardRef } from 'react';
import { Canvas, useThree, useFrame } from '@react-three/fiber';
import { OrbitControls, Grid, Line } from '@react-three/drei';
import * as THREE from 'three';
import { MdCenterFocusStrong } from 'react-icons/md';
import { useSelector } from 'react-redux';
import useUrdfRobot from '../hooks/useUrdfRobot';
import useJointStateSubscription from '../hooks/useJointStateSubscription';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

const CAMERA_PRESETS = {
  perspective: { label: 'Perspective' },
  front:       { label: 'Front' },
  side:        { label: 'Side' },
  top:         { label: 'Top' },
};

const MIN_USEFUL_MODEL_DIMENSION = 0.15;

const ROBOT_URDF_BASENAMES = {
  ffw_sg2_rev1: 'ffw_sg2_follower.urdf',
  ffw_sg2_rev2: 'ffw_sg2_follower.urdf',
  ffw_bg2_rev4: 'ffw_bg2_rev4_follower.urdf',
};

function getFallbackUrdfPath(robotType) {
  const basename = ROBOT_URDF_BASENAMES[robotType];
  return basename ? `/urdf/urdf/${basename}` : null;
}

function RobotModel({ robot }) {
  const groupRef = useRef();

  useEffect(() => {
    const group = groupRef.current;
    if (!robot || !group) return;
    group.add(robot);
    return () => {
      if (robot.parent === group) {
        group.remove(robot);
      }
    };
  }, [robot]);

  return <group ref={groupRef} rotation={[-Math.PI / 2, 0, 0]} />;
}

function TrajectoryPathLines({ paths }) {
  const convertedPaths = useMemo(() => {
    if (!paths) return { left: [], right: [] };

    const toArray = (pts) => pts.map((p) => [p.x, p.y, p.z]);

    return {
      left: paths.left?.length >= 2 ? toArray(paths.left) : [],
      right: paths.right?.length >= 2 ? toArray(paths.right) : [],
    };
  }, [paths]);

  return (
    <>
      {convertedPaths.left.length >= 2 && (
        <Line
          points={convertedPaths.left}
          color="#38bdf8"
          lineWidth={3}
          transparent
          opacity={0.7}
        />
      )}
      {convertedPaths.right.length >= 2 && (
        <Line
          points={convertedPaths.right}
          color="#f472b6"
          lineWidth={3}
          transparent
          opacity={0.7}
        />
      )}
      {convertedPaths.left.length > 0 && (
        <mesh position={convertedPaths.left[convertedPaths.left.length - 1]}>
          <sphereGeometry args={[0.015, 12, 12]} />
          <meshStandardMaterial color="#38bdf8" emissive="#38bdf8" emissiveIntensity={0.5} />
        </mesh>
      )}
      {convertedPaths.right.length > 0 && (
        <mesh position={convertedPaths.right[convertedPaths.right.length - 1]}>
          <sphereGeometry args={[0.015, 12, 12]} />
          <meshStandardMaterial color="#f472b6" emissive="#f472b6" emissiveIntensity={0.5} />
        </mesh>
      )}
    </>
  );
}

const CameraController = forwardRef(function CameraController({ robot }, ref) {
  const { camera } = useThree();
  const controlsRef = useRef();
  const centerRef = useRef(new THREE.Vector3());
  const baseDist = useRef(1);
  const initialized = useRef(false);

  useEffect(() => {
    if (!robot) return;
    initialized.current = false;

    // URDF meshes load asynchronously — poll until bounding box stabilizes
    let prevMaxDim = 0;
    let stableCount = 0;
    let attempts = 0;
    let appliedOnce = false;
    const MAX_ATTEMPTS = 40; // ~8 seconds max

    const poll = setInterval(() => {
      attempts++;
      robot.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(robot);
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);

      if (maxDim < MIN_USEFUL_MODEL_DIMENSION && attempts < MAX_ATTEMPTS) {
        if (attempts >= MAX_ATTEMPTS) clearInterval(poll);
        return;
      }

      // Check if size stabilized (< 1% change from previous)
      const changed = maxDim > 0 ? Math.abs(maxDim - prevMaxDim) / maxDim : 0;
      if (changed < 0.01 && prevMaxDim >= MIN_USEFUL_MODEL_DIMENSION) {
        stableCount++;
      } else {
        stableCount = 0;
      }
      prevMaxDim = maxDim;

      // Apply camera once stable for 2 consecutive checks, or on last attempt
      if (stableCount >= 2 || attempts >= MAX_ATTEMPTS) {
        const center = box.getCenter(new THREE.Vector3());
        // Shift focus upward to robot body center (60% height instead of 50%)
        center.y = box.min.y + size.y * 0.6;
        centerRef.current.copy(center);
        baseDist.current = Math.max(maxDim, MIN_USEFUL_MODEL_DIMENSION);
        initialized.current = true;
        applyPreset('perspective', false);
        appliedOnce = true;
      }

      if ((appliedOnce && stableCount >= 4) || attempts >= MAX_ATTEMPTS) {
        clearInterval(poll);
      }
    }, 200);

    return () => clearInterval(poll);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [robot]);

  const applyPreset = useCallback((presetName) => {
    if (!CAMERA_PRESETS[presetName]) return;

    const c = centerRef.current;
    const d = baseDist.current * 1.8;
    let pos;

    switch (presetName) {
      case 'front':
        pos = [c.x + d, c.y + d * 0.15, c.z];
        break;
      case 'side':
        pos = [c.x, c.y + d * 0.15, c.z + d];
        break;
      case 'top':
        pos = [c.x + 0.01, c.y + d * 1.2, c.z];
        break;
      case 'perspective':
      default:
        pos = [c.x + d * 0.7, c.y + d * 0.5, c.z - d * 0.7];
        break;
    }

    camera.position.set(...pos);
    camera.lookAt(c);
    camera.updateProjectionMatrix();
    if (controlsRef.current) {
      controlsRef.current.target.copy(c);
      controlsRef.current.update();
    }
  }, [camera]);

  useImperativeHandle(ref, () => ({
    applyPreset,
  }), [applyPreset]);

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.1}
      minDistance={0.3}
      maxDistance={10}
    />
  );
});

function SharedScene({ showGrid }) {
  return (
    <>
      <ambientLight intensity={0.6} />
      <directionalLight position={[5, 10, 5]} intensity={0.8} castShadow />
      <directionalLight position={[-3, 5, -3]} intensity={0.3} />
      {showGrid && (
        <Grid
          args={[10, 10]}
          cellSize={0.1}
          cellThickness={0.5}
          cellColor="#6b7280"
          sectionSize={0.5}
          sectionThickness={1}
          sectionColor="#374151"
          fadeDistance={5}
          fadeStrength={1}
          followCamera={false}
          infiniteGrid
        />
      )}
    </>
  );
}

function FrameInvalidator({ fps, active = true }) {
  const { invalidate } = useThree();

  useEffect(() => {
    if (!active) return undefined;
    const intervalMs = Math.max(16, Math.round(1000 / (fps || 30)));
    const id = setInterval(invalidate, intervalMs);
    return () => clearInterval(id);
  }, [active, fps, invalidate]);

  return null;
}

function SceneContent({ robot, trajectoryPaths, hasTrajectory, showGrid, cameraRef }) {
  return (
    <>
      <SharedScene showGrid={showGrid} />
      {robot && <RobotModel robot={robot} />}
      {hasTrajectory && <TrajectoryPathLines paths={trajectoryPaths} />}
      <CameraController ref={cameraRef} robot={robot} />
    </>
  );
}

function findJointFrame(timestamps, time) {
  if (!timestamps.length) return null;
  if (time <= timestamps[0]) return { lower: 0, upper: 0, alpha: 0 };

  const last = timestamps.length - 1;
  if (time >= timestamps[last]) return { lower: last, upper: last, alpha: 0 };

  let lo = 0;
  let hi = last;
  while (lo <= hi) {
    const mid = Math.floor((lo + hi) / 2);
    if (timestamps[mid] <= time) lo = mid + 1;
    else hi = mid - 1;
  }

  const upper = lo;
  const lower = Math.max(0, upper - 1);
  const t0 = timestamps[lower];
  const t1 = timestamps[upper];
  const alpha = t1 > t0 ? (time - t0) / (t1 - t0) : 0;
  return { lower, upper, alpha: Math.max(0, Math.min(1, alpha)) };
}

function applyInterpolatedJointFrame(robot, source, time) {
  if (!source?.timestamps?.length || !source?.names?.length || !source?.positions?.length) return false;

  const frame = findJointFrame(source.timestamps, time);
  if (!frame) return false;

  const numJoints = source.names.length;
  const lowerStart = frame.lower * numJoints;
  const upperStart = frame.upper * numJoints;

  source.names.forEach((name, j) => {
    const joint = robot.joints[name];
    if (!joint) return;

    const lowerValue = Number(source.positions[lowerStart + j] || 0);
    const upperValue = Number(source.positions[upperStart + j] || lowerValue);
    const value = lowerValue + (upperValue - lowerValue) * frame.alpha;
    joint.setJointValue(Number.isFinite(value) ? value : 0);
  });

  return true;
}

function ReplaySceneContent({
  robot,
  jointData,
  currentTime,
  isPlaying,
  playbackSpeed,
  showGrid,
  cameraRef,
}) {
  const { invalidate } = useThree();
  const lastAppliedTime = useRef(-1);
  const playbackClockRef = useRef({
    anchorTime: 0,
    anchorPerformanceTime: 0,
    isPlaying: false,
    playbackSpeed: 1,
  });

  useEffect(() => {
    playbackClockRef.current = {
      anchorTime: Number(currentTime) || 0,
      anchorPerformanceTime: performance.now(),
      isPlaying,
      playbackSpeed: Number(playbackSpeed) || 1,
    };
    invalidate();
  }, [currentTime, isPlaying, playbackSpeed, invalidate]);

  useFrame(() => {
    const source = jointData;
    if (!robot || !source?.timestamps?.length) return;

    const clock = playbackClockRef.current;
    const elapsed = clock.isPlaying
      ? ((performance.now() - clock.anchorPerformanceTime) / 1000) * clock.playbackSpeed
      : 0;
    const maxTime = source.timestamps[source.timestamps.length - 1];
    const displayTime = Math.max(0, Math.min(maxTime, clock.anchorTime + elapsed));

    if (Math.abs(displayTime - lastAppliedTime.current) < 1 / 120) return;
    if (applyInterpolatedJointFrame(robot, source, displayTime)) {
      lastAppliedTime.current = displayTime;
    }
  });

  return (
    <>
      <SharedScene showGrid={showGrid} />
      {robot && <RobotModel robot={robot} />}
      <CameraController ref={cameraRef} robot={robot} />
    </>
  );
}

function LoadingOverlay() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 z-10">
      <div className="flex flex-col items-center gap-2">
        <div className="w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
        <span className="text-white text-xs">Loading 3D Model...</span>
      </div>
    </div>
  );
}

function ErrorOverlay({ message, onRetry }) {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 z-10">
      <div className="flex flex-col items-center gap-2 text-center px-4">
        <span className="text-red-400 text-xs">{message}</span>
        {onRetry && (
          <button
            onClick={onRetry}
            className="text-xs text-blue-400 hover:text-blue-300 underline"
          >
            Retry
          </button>
        )}
      </div>
    </div>
  );
}

function canCreateWebGLContext() {
  if (typeof document === 'undefined') return false;
  try {
    const canvas = document.createElement('canvas');
    const context = (
      canvas.getContext('webgl2', { failIfMajorPerformanceCaveat: false })
      || canvas.getContext('webgl', { failIfMajorPerformanceCaveat: false })
      || canvas.getContext('experimental-webgl', { failIfMajorPerformanceCaveat: false })
    );
    if (context?.getExtension) {
      const loseContext = context.getExtension('WEBGL_lose_context');
      loseContext?.loseContext();
    }
    return Boolean(context);
  } catch {
    return false;
  }
}

function WebGLUnavailable() {
  return (
    <div className="flex h-full w-full items-center justify-center bg-gray-900 px-4 text-center">
      <div className="max-w-xs">
        <div className="text-sm font-semibold text-white">3D viewer unavailable</div>
        <div className="mt-2 text-xs leading-5 text-gray-400">
          This browser session could not create a WebGL context. Camera replay and joint data remain available.
        </div>
      </div>
    </div>
  );
}

function CameraPresetButtons({ onPreset, activePreset }) {
  return (
    <div className="absolute top-2 left-2 z-10 flex gap-1">
      {Object.entries(CAMERA_PRESETS).map(([key, preset]) => (
        <button
          key={key}
          onClick={() => onPreset(key)}
          className={`px-3 py-2 text-sm rounded font-medium transition-colors ${
            activePreset === key
              ? 'bg-blue-500 text-white'
              : 'bg-black/50 text-white/80 hover:bg-black/70'
          }`}
        >
          {preset.label}
        </button>
      ))}
    </div>
  );
}

function TrajectoryLegend({ visible }) {
  if (!visible) return null;
  return (
    <div className="absolute bottom-12 left-2 z-10 flex gap-3 bg-black/50 rounded px-2 py-1">
      <div className="flex items-center gap-1">
        <div className="w-4 h-1 rounded bg-sky-400" />
        <span className="text-xs text-white/70">Left</span>
      </div>
      <div className="flex items-center gap-1">
        <div className="w-4 h-1 rounded bg-pink-400" />
        <span className="text-xs text-white/70">Right</span>
      </div>
    </div>
  );
}

function SourceSelector({ visible, value, onChange }) {
  if (!visible) return null;

  return (
    <div className="absolute bottom-2 left-2 z-10 flex overflow-hidden rounded bg-black/50">
      {[
        ['state', 'State'],
        ['action', 'Action'],
      ].map(([key, label]) => (
        <button
          key={key}
          onClick={() => onChange(key)}
          className={`px-3 py-2 text-xs font-medium transition-colors ${
            value === key
              ? 'bg-emerald-500 text-white'
              : 'text-white/80 hover:bg-black/70'
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

export default function RobotViewer3D({
  mode = 'live',
  robotTypeOverride = '',
  jointData = null,
  currentTime = 0,
  isPlaying = false,
  playbackSpeed = 1,
  showGrid = true,
  className = '',
  showSourceSelector = false,
  liveUpdateHz = 15,
  defaultVisualizationSource = 'state',
}) {
  const globalRobotType = useSelector((state) => state.tasks.robotType);
  const robotType = robotTypeOverride || globalRobotType;
  const { getRobotInfo } = useRosServiceCaller();
  const [urdfPath, setUrdfPath] = useState(null);
  const [webglAvailable] = useState(() => canCreateWebGLContext());

  useEffect(() => {
    if (!webglAvailable || !robotType) {
      setUrdfPath(null);
      return;
    }
    const fallbackPath = getFallbackUrdfPath(robotType);
    setUrdfPath(fallbackPath);

    let cancelled = false;
    (async () => {
      try {
        const info = await getRobotInfo();
        if (cancelled || !info?.success || !info.urdf_path) return;
        // Orchestrator returns an absolute filesystem path under
        // shared/robot_configs/urdf/. nginx serves the same directory at
        // /urdf/urdf/ so map basename onto that prefix.
        const basename = info.urdf_path.split('/').pop();
        if (basename) setUrdfPath(`/urdf/urdf/${basename}`);
      } catch (e) {
        console.warn('Using local URDF fallback after robot info lookup failed:', e);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [robotType, getRobotInfo, webglAvailable]);

  const { robot, loading, error, setJointValues, computeTrajectoryPaths, reload } = useUrdfRobot(
    webglAvailable ? urdfPath : null
  );
  const cameraRef = useRef();
  const [activePreset, setActivePreset] = useState('perspective');
  const [visualizationSource, setVisualizationSource] = useState(defaultVisualizationSource);
  const [trajectoryPaths, setTrajectoryPaths] = useState(null);
  const [hasTrajectory, setHasTrajectory] = useState(false);
  const actionPreviewEnabled = showSourceSelector && visualizationSource === 'action';

  useEffect(() => {
    if (showSourceSelector) {
      setVisualizationSource(defaultVisualizationSource);
    }
  }, [defaultVisualizationSource, showSourceSelector]);

  const handleJointState = useCallback((data) => {
    setJointValues(data);
  }, [setJointValues]);

  const handleActionChunk = useCallback((data) => {
    if (!data?.names?.length || !data?.points?.length) return;

    const paths = computeTrajectoryPaths(data);
    if (paths.left.length > 0 || paths.right.length > 0) {
      setTrajectoryPaths(paths);
      setHasTrajectory(true);
    }
  }, [computeTrajectoryPaths]);

  useEffect(() => {
    if (!actionPreviewEnabled) {
      setTrajectoryPaths(null);
      setHasTrajectory(false);
    }
  }, [actionPreviewEnabled]);

  useJointStateSubscription(
    handleJointState,
    handleActionChunk,
    mode === 'live' && !!robot,
    {
      visualizationSource,
      liveUpdateHz,
      enableActionPreview: showSourceSelector,
    },
  );

  const handlePreset = useCallback((presetName) => {
    setActivePreset(presetName);
    cameraRef.current?.applyPreset(presetName);
  }, []);

  const canvasStyle = useMemo(() => ({
    background: 'linear-gradient(180deg, #1e293b 0%, #0f172a 100%)',
  }), []);

  const glRef = useRef(null);

  useEffect(() => {
    return () => {
      if (glRef.current) {
        glRef.current.dispose();
        glRef.current.forceContextLoss();
        glRef.current = null;
      }
    };
  }, []);

  return (
    <div className={`relative w-full h-full ${className}`}>
      {!webglAvailable ? (
        <WebGLUnavailable />
      ) : (
        <>
      {loading && <LoadingOverlay />}
      {error && <ErrorOverlay message={error} onRetry={reload} />}
      <CameraPresetButtons onPreset={handlePreset} activePreset={activePreset} />
      <TrajectoryLegend visible={hasTrajectory} />
      <SourceSelector
        visible={mode === 'live' && showSourceSelector}
        value={visualizationSource}
        onChange={setVisualizationSource}
      />
      <Canvas
        camera={{ position: [1.5, 1.5, 1.5], fov: 50, near: 0.01, far: 100 }}
        dpr={[1, 1.5]}
        style={canvasStyle}
        gl={{ antialias: false, alpha: false, powerPreference: 'low-power' }}
        frameloop="demand"
        onCreated={({ gl }) => {
          glRef.current = gl;
        }}
        shadows={{ type: THREE.PCFShadowMap }}
      >
        <FrameInvalidator
          fps={mode === 'replay' ? 15 : Math.max(10, liveUpdateHz || 15)}
          active={mode !== 'replay' || isPlaying}
        />
        {mode === 'replay' ? (
          <ReplaySceneContent
            robot={robot}
            jointData={jointData}
            currentTime={currentTime}
            isPlaying={isPlaying}
            playbackSpeed={playbackSpeed}
            showGrid={showGrid}
            cameraRef={cameraRef}
          />
        ) : (
          <SceneContent
            robot={robot}
            trajectoryPaths={trajectoryPaths}
            hasTrajectory={hasTrajectory}
            showGrid={showGrid}
            cameraRef={cameraRef}
          />
        )}
      </Canvas>
      <button
        onClick={() => handlePreset(activePreset)}
        className="absolute bottom-2 right-2 z-10 p-2.5 bg-black/50 text-white/80 rounded hover:bg-black/70 transition-colors"
        title="Reset camera"
      >
        <MdCenterFocusStrong size={22} />
      </button>
        </>
      )}
    </div>
  );
}
