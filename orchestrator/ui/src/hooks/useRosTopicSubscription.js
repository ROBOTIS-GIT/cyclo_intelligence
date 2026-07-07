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
// Author: Kiwoong Park

import { useRef, useEffect, useState, useCallback } from 'react';
import toast from 'react-hot-toast';
import { useDispatch, useSelector } from 'react-redux';
import ROSLIB from 'roslib';
import { RecordPhase, InferencePhase } from '../constants/taskPhases';
import {
  receiveServerRecordTaskInfo,
  setRecordStatus,
  setInferenceStatus,
  selectRobotType,
  setHeartbeatStatus,
  setLastHeartbeatTime,
  setJoystickMode,
  setRecordingMonitor,
  setCameraRecordingMonitor,
} from '../features/tasks/taskSlice';
import {
  setIsTraining,
  setTopicReceived,
  setTrainingInfo,
  setCurrentStep,
  setLastUpdate,
  setSelectedUser,
  setSelectedDataset,
  setCurrentLoss,
} from '../features/training/trainingSlice';
import {
  setHFStatus,
  setDownloadStatus,
  setHFUserId,
  setHFRepoIdUpload,
  setHFRepoIdDownload,
  setUploadStatus,
  setConversionStatus,
} from '../features/editDataset/editDatasetSlice';
import HFStatus from '../constants/HFStatus';
import PageType from '../constants/pageType';
import store from '../store/store';
import rosConnectionManager from '../utils/rosConnectionManager';
import {
  hasRosTaskInfoPayload,
  rosTaskInfoToUiTaskInfo,
  shouldApplyServerTaskInfoToPage,
} from '../utils/taskInfoSync';
import {
  buildCameraMonitorTopics,
  isMonitorOnlyStatusMessage,
  shouldAnnounceRecordingStart,
} from '../utils/recordingStatusMonitor';

export function useRosTopicSubscription() {
  const recordingStatusTopicRef = useRef(null);
  const inferenceStatusTopicRef = useRef(null);
  const dataStatusTopicRef = useRef(null);
  const heartbeatTopicRef = useRef(null);
  const trainingStatusTopicRef = useRef(null);
  const previousRecordPhaseRef = useRef(null);
  const hasSeenRecordPhaseRef = useRef(false);
  const hfStatusTopicRef = useRef(null);
  const actionEventTopicRef = useRef(null);
  const recordingMonitorTopicRef = useRef(null);
  const joystickModeTopicRef = useRef(null);
  // HOME only seeds task_info once. Record/Inference pages accept later
  // backend echoes so multiple browser tabs stay in sync.
  const initialTaskInfoSyncRef = useRef(false);
  // Dedup inference-status error toasts so a sticky error field doesn't spam.
  const previousInferenceErrorRef = useRef('');
  const previousCameraMonitorIssueKeyRef = useRef('');
  const lastCameraMonitorAlertAtRef = useRef(0);

  const dispatch = useDispatch();
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);
  const [connected, setConnected] = useState(false);

  const preferredVoiceRef = useRef(null);

  useEffect(() => {
    if (!('speechSynthesis' in window)) return;
    const pickVoice = () => {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length === 0) return;
      preferredVoiceRef.current = voices.find(v =>
        v.lang.startsWith('en') && v.name.toLowerCase().includes('female')
      ) || voices.find(v =>
        v.lang.startsWith('en') && /samantha|karen|victoria|fiona|moira|tessa/i.test(v.name)
      ) || voices.find(v =>
        v.lang.startsWith('en-US') && !v.name.toLowerCase().includes('male')
      ) || null;
      if (preferredVoiceRef.current) {
        console.log('TTS voice selected:', preferredVoiceRef.current.name);
      }
    };
    pickVoice();
    window.speechSynthesis.addEventListener('voiceschanged', pickVoice);
    return () => window.speechSynthesis.removeEventListener('voiceschanged', pickVoice);
  }, []);

  const speakText = useCallback(
    (text, lang = 'en-US') => {
      try {
        if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel();
          window.speechSynthesis.resume();
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = lang;
          utterance.rate = 1.1;
          utterance.pitch = 1.1;
          utterance.volume = 1.0;
          if (lang === 'en-US' && preferredVoiceRef.current) {
            utterance.voice = preferredVoiceRef.current;
          }
          window.speechSynthesis.speak(utterance);
          console.log(`Speech: "${text}"`);
        } else {
          console.warn('Speech synthesis not available');
        }
      } catch (error) {
        console.warn('Speech failed:', error);
      }
    },
    []
  );

  const notifyCameraMonitorIssues = useCallback(
    (cameraTopics) => {
      const issueRows = cameraTopics.filter((topic) => topic.status !== 0);
      const issueKey = issueRows
        .map((topic) => `${topic.name}:${topic.status}:${topic.timestampStatus}`)
        .sort()
        .join('|');

      if (!issueKey) {
        previousCameraMonitorIssueKeyRef.current = '';
        return;
      }

      const now = Date.now();
      const ALERT_COOLDOWN_MS = 10000;
      if (
        issueKey === previousCameraMonitorIssueKeyRef.current ||
        now - lastCameraMonitorAlertAtRef.current < ALERT_COOLDOWN_MS
      ) {
        return;
      }

      const summary = issueRows
        .slice(0, 3)
        .map((topic) => {
          const skewText = topic.timestampStatus !== 0
            ? `, skew ${topic.timestampSkewS.toFixed(2)}s`
            : '';
          return `${topic.name}: ${topic.statusLabel}${skewText}`;
        })
        .join('\n');
      const moreText = issueRows.length > 3
        ? `\n+${issueRows.length - 3} more`
        : '';

      toast.error(`Camera monitor warning\n${summary}${moreText}`, {
        duration: 9000,
      });
      speakText('camera warning');
      previousCameraMonitorIssueKeyRef.current = issueKey;
      lastCameraMonitorAlertAtRef.current = now;
    },
    [speakText]
  );

  // Helper function to unsubscribe from a topic
  const unsubscribeFromTopic = useCallback((topicRef, topicName) => {
    if (topicRef.current) {
      topicRef.current.unsubscribe();
      topicRef.current = null;
      console.log(`${topicName} topic unsubscribed`);
    }
  }, []);

  const cleanup = useCallback(() => {
    console.log('Starting ROS subscriptions cleanup...');

    // Unsubscribe from all topics
    unsubscribeFromTopic(recordingStatusTopicRef, 'Recording status');
    unsubscribeFromTopic(inferenceStatusTopicRef, 'Inference status');
    unsubscribeFromTopic(dataStatusTopicRef, 'Data operation status');
    unsubscribeFromTopic(heartbeatTopicRef, 'Heartbeat');
    unsubscribeFromTopic(trainingStatusTopicRef, 'Training status');
    unsubscribeFromTopic(hfStatusTopicRef, 'HF status');
    unsubscribeFromTopic(actionEventTopicRef, 'Action event');
    unsubscribeFromTopic(recordingMonitorTopicRef, 'Recording monitor');
    unsubscribeFromTopic(joystickModeTopicRef, 'Joystick mode');

    // Reset transition trackers
    previousRecordPhaseRef.current = null;
    hasSeenRecordPhaseRef.current = false;
    previousInferenceErrorRef.current = '';
    previousCameraMonitorIssueKeyRef.current = '';
    lastCameraMonitorAlertAtRef.current = 0;
    initialTaskInfoSyncRef.current = false;

    setConnected(false);
    dispatch(setHeartbeatStatus('disconnected'));
    console.log('ROS task status cleanup completed');
  }, [dispatch, unsubscribeFromTopic]);

  const subscribeToRecordingStatus = useCallback(async () => {
    try {
      const VOICE_DELAY = 100;

      const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
      if (!ros) return;

      if (recordingStatusTopicRef.current) {
        console.log('Recording status already subscribed, skipping...');
        return;
      }

      setConnected(true);
      recordingStatusTopicRef.current = new ROSLIB.Topic({
        ros,
        name: '/data/recording/status',
        messageType: 'interfaces/msg/RecordingStatus',
      });

      recordingStatusTopicRef.current.subscribe((msg) => {
        const currentPhase = msg.record_phase;
        const previousPhase = previousRecordPhaseRef.current;
        const cameraTopics = buildCameraMonitorTopics(msg);
        const monitorOnlyMessage = isMonitorOnlyStatusMessage(msg, cameraTopics);

        if (
          !monitorOnlyMessage &&
          shouldAnnounceRecordingStart({
            hasSeenRecordPhase: hasSeenRecordPhaseRef.current,
            previousPhase,
            currentPhase,
            proceedTime: msg.proceed_time,
          })
        ) {
          console.log('Recording started - speaking notification');
          setTimeout(() => {
            speakText('Recording started');
          }, VOICE_DELAY);
          toast.success('Recording started!');
        }
        if (!monitorOnlyMessage) {
          previousRecordPhaseRef.current = currentPhase;
          hasSeenRecordPhaseRef.current = true;
        }

        // SAVING (post-record encode) reports progress via encoding_progress.
        // Conversion progress is on /data/status (DataOperationStatus,
        // OP_CONVERSION) routed through editDatasetSlice.conversionStatus.
        const encodingProgress =
          currentPhase === RecordPhase.SAVING ? (msg.encoding_progress || 0) : 0;
        const recordingWarnings = Array.isArray(msg.recording_warnings)
          ? msg.recording_warnings.filter((warning) => !!warning)
          : [];
        dispatch(setCameraRecordingMonitor(cameraTopics));
        notifyCameraMonitorIssues(cameraTopics);

        const isRunning =
          currentPhase === RecordPhase.RECORDING ||
          currentPhase === RecordPhase.SAVING;
        // Adopt robot_type as the global value when present.
        if (msg.robot_type && msg.robot_type.trim() !== '') {
          const tasksState = store.getState().tasks;
          const shouldUpdateRobotType =
            msg.robot_type !== tasksState.robotType ||
            Boolean(tasksState.robotTypeStatusGuardUntilMs);
          if (shouldUpdateRobotType) {
            dispatch(selectRobotType({
              robotType: msg.robot_type,
              source: 'status',
              receivedAtMs: Date.now(),
            }));
          }
        }

        if (!monitorOnlyMessage) {
          dispatch(
            setRecordStatus({
              taskName: msg.task_info?.task_name || 'idle',
              running: isRunning,
              recordPhase: currentPhase || 0,
              progress: Math.round(encodingProgress),
              encodingProgress,
              proceedTime: msg.proceed_time || 0,
              currentEpisodeNumber: msg.current_episode_number || 0,
              currentScenarioNumber: msg.current_scenario_number || 0,
              currentTaskInstruction: msg.current_task_instruction || '',
              currentSubtaskIndex: msg.current_subtask_index || 0,
              subtaskCount: msg.subtask_count || 0,
              currentSubtaskInstruction: msg.current_subtask_instruction || '',
              subtaskInstructions: msg.subtask_instructions || [],
              savedSubtaskIndices: Array.isArray(msg.saved_subtask_indices)
                ? msg.saved_subtask_indices.map((idx) => Number(idx))
                : null,
              userId: msg.task_info?.user_id || '',
              usedStorageSize: msg.used_storage_size || 0,
              totalStorageSize: msg.total_storage_size || 0,
              usedCpu: msg.used_cpu || 0,
              usedRamSize: msg.used_ram_size || 0,
              totalRamSize: msg.total_ram_size || 0,
              recordingWarnings,
              topicReceived: true,
            })
          );
        }

        // Keep Record-page task information aligned with the server echo,
        // unless this browser is protecting an in-progress local draft.
        const uiTaskInfo = hasRosTaskInfoPayload(msg.task_info)
          ? rosTaskInfoToUiTaskInfo(msg.task_info)
          : null;
        const currentPage = store.getState().ui.currentPage;
        const currentInferencePhase = store.getState().tasks.inferenceStatus.inferencePhase;
        const shouldApplyTaskInfo = shouldApplyServerTaskInfoToPage({
          taskInfo: uiTaskInfo,
          currentPage,
          inferencePhase: currentInferencePhase,
          initialTaskInfoSynced: initialTaskInfoSyncRef.current,
        });
        if (uiTaskInfo && shouldApplyTaskInfo) {
          if (currentPage === PageType.HOME) {
            initialTaskInfoSyncRef.current = true;
          }
          dispatch(receiveServerRecordTaskInfo(uiTaskInfo));
        }
      });
    } catch (error) {
      console.error('Failed to subscribe to recording status topic:', error);
    }
  }, [
    dispatch,
    rosbridgeUrl,
    speakText,
    notifyCameraMonitorIssues,
  ]);

  const subscribeToDataStatus = useCallback(async () => {
    try {
      const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
      if (!ros) return;
      if (dataStatusTopicRef.current) {
        console.log('Data operation status already subscribed, skipping...');
        return;
      }

      // DataOperationStatus enum values — must match
      // interfaces/msg/DataOperationStatus.msg.
      const OP_RECORDING = 0;
      const OP_CONVERSION = 1;
      const STATUS_NAMES = {
        0: 'idle',
        1: 'running',
        2: 'completed',
        3: 'failed',
        4: 'cancelled',
      };

      dataStatusTopicRef.current = new ROSLIB.Topic({
        ros,
        name: '/data/status',
        messageType: 'interfaces/msg/DataOperationStatus',
      });

      dataStatusTopicRef.current.subscribe((msg) => {
        if (msg.operation_type === OP_RECORDING) {
          dispatch(
            setRecordStatus({
              recordingOperationStatus: STATUS_NAMES[msg.status] || 'idle',
              recordingOperationStage: msg.stage || '',
              recordingOperationMessage: msg.message || '',
            })
          );
          return;
        }

        if (msg.operation_type !== OP_CONVERSION) {
          // OP_HF has its own /huggingface/status feed; OP_EDIT is not
          // surfaced as live progress in the UI today.
          return;
        }
        dispatch(
          setConversionStatus({
            status: STATUS_NAMES[msg.status] || 'idle',
            progress: msg.progress_percentage || 0,
            stage: msg.stage || '',
            message: msg.message || '',
            jobId: msg.job_id || '',
          })
        );
      });
      console.log('Data operation status subscription established');
    } catch (error) {
      console.error('Failed to subscribe to data status topic:', error);
    }
  }, [dispatch, rosbridgeUrl]);

  const subscribeToInferenceStatus = useCallback(async () => {
    try {
      const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
      if (!ros) return;

      if (inferenceStatusTopicRef.current) {
        console.log('Inference status already subscribed, skipping...');
        return;
      }

      setConnected(true);
      inferenceStatusTopicRef.current = new ROSLIB.Topic({
        ros,
        name: '/task/inference_status',
        messageType: 'interfaces/msg/InferenceStatus',
      });

      inferenceStatusTopicRef.current.subscribe((msg) => {
        // Show the toast once when a new error appears, but DO NOT bail out:
        // we still need to dispatch setInferenceStatus below so the phase
        // update (e.g. backend resetting LOADING → READY after a failed
        // setup) is reflected in the UI.
        if (msg.error && msg.error !== previousInferenceErrorRef.current) {
          console.log('error:', msg.error);
          toast.error(msg.error);
        }
        previousInferenceErrorRef.current = msg.error || '';

        if (msg.robot_type && msg.robot_type.trim() !== '') {
          const tasksState = store.getState().tasks;
          const shouldUpdateRobotType =
            msg.robot_type !== tasksState.robotType ||
            Boolean(tasksState.robotTypeStatusGuardUntilMs);
          if (shouldUpdateRobotType) {
            dispatch(selectRobotType({
              robotType: msg.robot_type,
              source: 'status',
              receivedAtMs: Date.now(),
            }));
          }
        }

        dispatch(
          setInferenceStatus({
            inferencePhase: msg.inference_phase || 0,
            error: msg.error || '',
            topicReceived: true,
          })
        );
      });
    } catch (error) {
      console.error('Failed to subscribe to inference status topic:', error);
    }
  }, [dispatch, rosbridgeUrl]);

  const subscribeToActionEvent = useCallback(async () => {
    try {
      const VOICE_DELAY = 100;
      const ACTION_VOICE_MAP = {
        start: 'Recording started',
        finish: 'Recording finished',
        cancel: 'Cancelled',
        review_on: 'Previous data needs review',
        review_off: 'Previous data review cleared',
      };

      const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
      if (!ros) return;

      if (actionEventTopicRef.current) {
        console.log('Action event already subscribed, skipping...');
        return;
      }

      actionEventTopicRef.current = new ROSLIB.Topic({
        ros,
        name: '/task/action_event',
        messageType: 'std_msgs/msg/String',
      });

      actionEventTopicRef.current.subscribe((msg) => {
        const action = msg.data;
        console.log('Received action event:', action);

        const voiceText = ACTION_VOICE_MAP[action];
        if (voiceText) {
          setTimeout(() => {
            speakText(voiceText);
          }, VOICE_DELAY);
        }
      });

      console.log('Action event subscription established');
    } catch (error) {
      console.error('Failed to subscribe to action event topic:', error);
    }
  }, [rosbridgeUrl, speakText]);

  const subscribeToHeartbeat = useCallback(async () => {
    try {
      const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
      if (!ros) return;

      // Skip if already subscribed
      if (heartbeatTopicRef.current) {
        console.log('Heartbeat already subscribed, skipping...');
        return;
      }

      heartbeatTopicRef.current = new ROSLIB.Topic({
        ros,
        name: '/heartbeat',
        messageType: 'std_msgs/msg/Empty',
      });

      heartbeatTopicRef.current.subscribe(() => {
        dispatch(setHeartbeatStatus('connected'));
        dispatch(setLastHeartbeatTime(Date.now()));
      });

      console.log('Heartbeat subscription established');
    } catch (error) {
      console.error('Failed to subscribe to heartbeat topic:', error);
    }
  }, [dispatch, rosbridgeUrl]);

  // Helper: human-readable phase names
  const getRecordPhaseName = useCallback((phase) => ({
    [RecordPhase.READY]: 'READY',
    [RecordPhase.RECORDING]: 'RECORDING',
    [RecordPhase.SAVING]: 'SAVING',
    [RecordPhase.PAUSED]: 'PAUSED',
  }[phase] || 'UNKNOWN'), []);

  const getInferencePhaseName = useCallback((phase) => ({
    [InferencePhase.READY]: 'READY',
    [InferencePhase.LOADING]: 'LOADING',
    [InferencePhase.INFERENCING]: 'INFERENCING',
    [InferencePhase.PAUSED]: 'PAUSED',
  }[phase] || 'UNKNOWN'), []);

  const subscribeToTrainingStatus = useCallback(async () => {
    try {
      const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
      if (!ros) return;

      // Skip if already subscribed
      if (trainingStatusTopicRef.current) {
        console.log('Training status already subscribed, skipping...');
        return;
      }

      setConnected(true);
      trainingStatusTopicRef.current = new ROSLIB.Topic({
        ros,
        name: '/training/status',
        messageType: 'interfaces/msg/TrainingStatus',
      });

      trainingStatusTopicRef.current.subscribe((msg) => {
        console.log('Received training status:', msg);

        if (msg.error !== '') {
          console.log('error:', msg.error);
          toast.error(msg.error);
          return;
        }

        // ROS message to React state
        dispatch(
          setTrainingInfo({
            datasetRepoId: msg.training_info.dataset || '',
            policyType: msg.training_info.policy_type || '',
            policyDevice: msg.training_info.policy_device || '',
            outputFolderName: msg.training_info.output_folder_name || '',
            resume: msg.training_info.resume || false,
            seed: msg.training_info.seed || 0,
            numWorkers: msg.training_info.num_workers || 0,
            batchSize: msg.training_info.batch_size || 0,
            steps: msg.training_info.steps || 0,
            evalFreq: msg.training_info.eval_freq || 0,
            logFreq: msg.training_info.log_freq || 0,
            saveFreq: msg.training_info.save_freq || 0,
          })
        );

        const datasetParts = msg.training_info.dataset.split('/');
        dispatch(setSelectedUser(datasetParts[0] || ''));
        dispatch(setSelectedDataset(datasetParts[1] || ''));
        dispatch(setIsTraining(msg.is_training));
        dispatch(setCurrentStep(msg.current_step || 0));
        dispatch(setCurrentLoss(msg.current_loss));
        dispatch(setTopicReceived(true));
        dispatch(setLastUpdate(Date.now()));
      });
    } catch (error) {
      console.error('Failed to subscribe to training status topic:', error);
    }
  }, [dispatch, rosbridgeUrl]);

  const subscribeHFStatus = useCallback(async () => {
    try {
      const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
      if (!ros) return;

      // Skip if already subscribed
      if (hfStatusTopicRef.current) {
        console.log('HF status already subscribed, skipping...');
        return;
      }

      hfStatusTopicRef.current = new ROSLIB.Topic({
        ros,
        name: '/huggingface/status',
        messageType: 'interfaces/msg/HFOperationStatus',
      });

      hfStatusTopicRef.current.subscribe((msg) => {
        console.log('Received HF status:', msg);

        const status = msg.status;
        const operation = msg.operation;
        const repoId = msg.repo_id;
        // const localPath = msg.local_path;
        const message = msg.message;
        const progressCurrent = msg.progress_current;
        const progressTotal = msg.progress_total;
        const progressPercentage = msg.progress_percentage;

        if (status === 'Failed') {
          toast.error(message);
        } else if (status === 'Success') {
          toast.success(message);
        }

        console.log('status:', status);

        // Check the current status from the store
        const currentStatus = store.getState().editDataset.hfStatus;

        if (
          (currentStatus === HFStatus.SUCCESS || currentStatus === HFStatus.FAILED) &&
          status === HFStatus.IDLE
        ) {
          console.log('Maintaining SUCCESS status, skipping IDLE update');
          // Skip updating the status
        } else {
          console.log('Updating HF status to:', status);
          dispatch(setHFStatus(status));
        }

        if (operation === 'upload') {
          dispatch(
            setUploadStatus({
              current: progressCurrent,
              total: progressTotal,
              percentage: progressPercentage.toFixed(2),
            })
          );
        } else if (operation === 'download') {
          dispatch(
            setDownloadStatus({
              current: progressCurrent,
              total: progressTotal,
              percentage: progressPercentage.toFixed(2),
            })
          );
        }
        const userId = repoId.split('/')[0];
        const repoName = repoId.split('/')[1];

        if (userId?.trim() && repoName?.trim()) {
          dispatch(setHFUserId(userId));

          if (operation === 'upload') {
            dispatch(setHFRepoIdUpload(repoName));
          } else if (operation === 'download') {
            dispatch(setHFRepoIdDownload(repoName));
          }
        }
      });

      console.log('HF status subscription established');
    } catch (error) {
      console.error('Failed to subscribe to HF status topic:', error);
    }
  }, [dispatch, rosbridgeUrl]);

  // Per-topic recording monitor (1 Hz while recording is active).
  const subscribeToRecordingMonitor = useCallback(async () => {
    try {
      const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
      if (!ros) return;
      if (recordingMonitorTopicRef.current) return;

      recordingMonitorTopicRef.current = new ROSLIB.Topic({
        ros,
        name: '/rosbag_recorder/monitor',
        messageType: 'rosbag_recorder/msg/RecordingMonitor',
      });

      recordingMonitorTopicRef.current.subscribe((msg) => {
        // rosbridge serialises uint8[] as a base64 string; decode it back
        // to a plain numeric array so status[i] gives 0/1/2 not a character.
        let statusArr = msg.status || [];
        if (typeof statusArr === 'string') {
          const bin = atob(statusArr);
          statusArr = Array.from(bin, (ch) => ch.charCodeAt(0));
        }

        const topics = (msg.topic_names || []).map((name, i) => ({
          name,
          rateHz: msg.rates_hz?.[i] ?? 0,
          baselineHz: msg.baseline_hz?.[i] ?? 0,
          secondsSinceLast: msg.seconds_since_last?.[i] ?? -1,
          status: statusArr[i] ?? 0,
        }));
        dispatch(setRecordingMonitor({
          topics,
          totalReceived: msg.total_received || 0,
          totalWritten: msg.total_written || 0,
        }));
      });

      console.log('Recording monitor subscription established');
    } catch (error) {
      console.error('Failed to subscribe to recording monitor topic:', error);
    }
  }, [dispatch, rosbridgeUrl]);

  // Leader joystick operating mode (fires only on button press, not continuous).
  const subscribeToJoystickMode = useCallback(async () => {
    try {
      const ros = await rosConnectionManager.getConnection(rosbridgeUrl);
      if (!ros) return;
      if (joystickModeTopicRef.current) return;

      joystickModeTopicRef.current = new ROSLIB.Topic({
        ros,
        name: '/leader/joystick_controller_right/joystick_mode',
        messageType: 'std_msgs/msg/String',
      });

      joystickModeTopicRef.current.subscribe((msg) => {
        dispatch(setJoystickMode(msg.data));
      });

      console.log('Joystick mode subscription established');
    } catch (error) {
      console.error('Failed to subscribe to joystick mode topic:', error);
    }
  }, [dispatch, rosbridgeUrl]);

  // Manual initialization function
  const initializeSubscriptions = useCallback(async () => {
    if (!rosbridgeUrl) {
      console.warn('Cannot initialize subscriptions: rosbridgeUrl is not set');
      return;
    }

    console.log('Manually initializing ROS subscriptions...');

    // Cleanup previous subscriptions before creating new ones
    cleanup();

    try {
      await subscribeToRecordingStatus();
      await subscribeToInferenceStatus();
      await subscribeToDataStatus();
      await subscribeToHeartbeat();
      await subscribeToActionEvent();
      await subscribeToTrainingStatus();
      await subscribeHFStatus();
      await subscribeToRecordingMonitor();
      await subscribeToJoystickMode();
      console.log('ROS subscriptions initialized successfully');
    } catch (error) {
      console.error('Failed to initialize ROS subscriptions:', error);
    }
  }, [
    rosbridgeUrl,
    cleanup,
    subscribeToRecordingStatus,
    subscribeToInferenceStatus,
    subscribeToDataStatus,
    subscribeToHeartbeat,
    subscribeToActionEvent,
    subscribeToTrainingStatus,
    subscribeHFStatus,
    subscribeToRecordingMonitor,
    subscribeToJoystickMode,
  ]);

  // Auto-start connection and subscription
  useEffect(() => {
    if (!rosbridgeUrl) return;

    initializeSubscriptions();

    return cleanup;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rosbridgeUrl]); // Only rosbridgeUrl as dependency to prevent unnecessary re-subscriptions

  return {
    connected,
    subscribeToRecordingStatus,
    subscribeToInferenceStatus,
    cleanup,
    getRecordPhaseName,
    getInferencePhaseName,
    subscribeToTrainingStatus,
    subscribeHFStatus,
    initializeSubscriptions, // Manual initialization function
  };
}
