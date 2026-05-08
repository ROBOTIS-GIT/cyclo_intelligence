import { useState, useEffect, useRef, useCallback } from 'react';
import * as THREE from 'three';
import URDFLoader from 'urdf-loader';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader';

const DEFAULT_URDF_PATH = '/urdf/urdf/ffw_sg2_follower.urdf';
const EE_LINKS = ['end_effector_l_link', 'end_effector_r_link'];
const ROS_TO_THREE = new THREE.Matrix4().makeRotationX(-Math.PI / 2);

function createFkSolver(robot) {
  const solver = robot.clone();

  const jointMap = {};
  solver.traverse((child) => {
    if (child.isURDFJoint) {
      const src = robot.joints[child.name];
      if (src) {
        child.jointType = src.jointType;
        child.axis = src.axis?.clone ? src.axis.clone() : src.axis;
        child.limit = { ...src.limit };
        child.ignoreLimits = src.ignoreLimits;
        child.jointValue = [...(src.jointValue || [])];
        child.mimicJoints = src.mimicJoints ? [...src.mimicJoints] : [];
      }
      jointMap[child.name] = child;
    }
  });
  solver.joints = jointMap;

  return solver;
}

export default function useUrdfRobot(urdfPath = DEFAULT_URDF_PATH) {
  const [robot, setRobot] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const robotRef = useRef(null);
  const fkSolverRef = useRef(null);

  const loadRobot = useCallback(() => {
    setLoading(true);
    setError(null);

    const loader = new URDFLoader();
    const stlLoader = new STLLoader();

    loader.packages = {
      'ffw_description': '/urdf/ffw_description',
    };

    const makeFallback = (color = 0x888888) => new THREE.Mesh(
      new THREE.BoxGeometry(0.02, 0.02, 0.02),
      new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.3 })
    );

    const loadStl = (url, onComplete) => {
      stlLoader.load(
        url,
        (geometry) => {
          geometry.computeVertexNormals();
          const material = new THREE.MeshStandardMaterial({
            color: 0xcccccc,
            metalness: 0.3,
            roughness: 0.6,
          });
          onComplete(new THREE.Mesh(geometry, material));
        },
        undefined,
        () => {
          console.warn('Failed to load mesh:', url);
          onComplete(makeFallback(0xff4444));
        }
      );
    };

    loader.loadMeshCb = (path, _manager, onComplete) => {
      if (path.startsWith('package://')) {
        onComplete(makeFallback());
        return;
      }

      // urdf-loader prepends base URL to file:/// paths, producing
      // e.g. /urdf/urdf/file:///root/ros2_ws/install/<pkg>/share/<pkg>/meshes/...
      const fileIdx = path.indexOf('file:///');
      if (fileIdx !== -1) {
        const absPath = path.substring(fileIdx + 'file:///'.length);
        const shareMatch = absPath.match(/\/share\/([^/]+)\/(.+)$/);
        if (shareMatch) {
          const [, pkgName, relPath] = shareMatch;
          const pkgBase = loader.packages[pkgName];
          if (pkgBase) {
            loadStl(`${pkgBase}/${relPath}`, onComplete);
            return;
          }
        }
        onComplete(makeFallback());
        return;
      }

      loadStl(path, onComplete);
    };

    loader.load(
      urdfPath,
      (loadedRobot) => {
        loadedRobot.traverse((child) => {
          if (child.isMesh && child.material) {
            const urdfMaterial = child.userData?.urdfMaterial;
            if (urdfMaterial?.color) {
              child.material.color.setRGB(
                urdfMaterial.color.r,
                urdfMaterial.color.g,
                urdfMaterial.color.b
              );
              if (urdfMaterial.color.a !== undefined && urdfMaterial.color.a < 1) {
                child.material.transparent = true;
                child.material.opacity = urdfMaterial.color.a;
              }
            }
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });

        robotRef.current = loadedRobot;
        setRobot(loadedRobot);

        fkSolverRef.current = createFkSolver(loadedRobot);

        setLoading(false);
      },
      undefined,
      (err) => {
        console.error('URDF load error:', err);
        setError(err?.message || 'Failed to load URDF');
        setLoading(false);
      }
    );
  }, [urdfPath]);

  useEffect(() => {
    loadRobot();
    return () => {
      if (robotRef.current) {
        robotRef.current.traverse((child) => {
          if (child.isMesh) {
            child.geometry?.dispose();
            if (child.material) {
              if (Array.isArray(child.material)) {
                child.material.forEach((m) => m.dispose());
              } else {
                child.material.dispose();
              }
            }
          }
        });
        robotRef.current = null;
      }
      fkSolverRef.current = null;
    };
  }, [loadRobot]);

  const setJointValues = useCallback((jointData) => {
    const r = robotRef.current;
    if (!r) return;
    if (Array.isArray(jointData.name)) {
      jointData.name.forEach((name, i) => {
        if (r.joints[name]) {
          r.joints[name].setJointValue(jointData.position[i]);
        }
      });
    } else if (typeof jointData === 'object') {
      Object.entries(jointData).forEach(([name, value]) => {
        if (r.joints[name]) {
          r.joints[name].setJointValue(value);
        }
      });
    }
  }, []);

  const computeTrajectoryPaths = useCallback((trajectoryData) => {
    const fk = fkSolverRef.current;
    const r = robotRef.current;
    if (!fk || !r || !trajectoryData?.names || !trajectoryData?.points?.length) {
      return { left: [], right: [] };
    }

    Object.keys(r.joints).forEach((name) => {
      if (fk.joints[name] && r.joints[name]) {
        const val = r.joints[name].jointValue;
        if (val && val.length > 0) {
          fk.joints[name].setJointValue(...val);
        }
      }
    });

    const eeObjects = {};
    fk.traverse((child) => {
      if (EE_LINKS.includes(child.name)) {
        eeObjects[child.name] = child;
      }
    });

    if (!eeObjects[EE_LINKS[0]] && !eeObjects[EE_LINKS[1]]) {
      return { left: [], right: [] };
    }

    const leftPath = [];
    const rightPath = [];
    const localPos = new THREE.Vector3();

    for (const positions of trajectoryData.points) {
      trajectoryData.names.forEach((name, i) => {
        if (fk.joints[name]) {
          fk.joints[name].setJointValue(positions[i]);
        }
      });

      fk.updateMatrixWorld(true);

      if (eeObjects[EE_LINKS[0]]) {
        eeObjects[EE_LINKS[0]].getWorldPosition(localPos);
        leftPath.push(localPos.clone().applyMatrix4(ROS_TO_THREE));
      }
      if (eeObjects[EE_LINKS[1]]) {
        eeObjects[EE_LINKS[1]].getWorldPosition(localPos);
        rightPath.push(localPos.clone().applyMatrix4(ROS_TO_THREE));
      }
    }

    return { left: leftPath, right: rightPath };
  }, []);

  return { robot, loading, error, setJointValues, computeTrajectoryPaths, reload: loadRobot };
}
