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
  setRecordStatus,
  setInferenceStatus,
  setTaskInfo,
  selectRobotType,
  setHeartbeatStatus,
  setLastHeartbeatTime,
  setJoystickMode,
  setRecordingMonitor,
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
import store from '../store/store';
import rosConnectionManager from '../utils/rosConnectionManager';

export function useRosTopicSubscription() {
  const recordingStatusTopicRef = useRef(null);
  const inferenceStatusTopicRef = useRef(null);
  const dataStatusTopicRef = useRef(null);
  const heartbeatTopicRef = useRef(null);
  const trainingStatusTopicRef = useRef(null);
  const previousRecordPhaseRef = useRef(null);
  const audioContextRef = useRef(null);
  const hfStatusTopicRef = useRef(null);
  const actionEventTopicRef = useRef(null);
  const recordingMonitorTopicRef = useRef(null);
  const joystickModeTopicRef = useRef(null);
  // One-shot guard so the backend's task_info echo only seeds redux on the
  // first message; subsequent echoes would clobber whatever the user is
  // currently typing in InfoPanel.
  const initialTaskInfoSyncRef = useRef(false);
  // Dedup inference-status error toasts so a sticky error field doesn't spam.
  const previousInferenceErrorRef = useRef('');

  const dispatch = useDispatch();
  const rosbridgeUrl = useSelector((state) => state.ros.rosbridgeUrl);
  const [connected, setConnected] = useState(false);

  const initializeAudioContext = useCallback(() => {
    if (!audioContextRef.current) {
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioContextRef.current;
  }, []);

  const playBeep = useCallback(
    async (frequency = 1000, duration = 400) => {
      const INITIAL_GAIN = 1.0;
      const FINAL_GAIN = 0.01;
      const FALLBACK_VIBRATION_PATTERN = [200, 100, 200];

      try {
        const audioContext = initializeAudioContext();

        if (audioContext.state === 'suspended') {
          await audioContext.resume();
        }

        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);

        oscillator.frequency.value = frequency;
        oscillator.type = 'sine';

        gainNode.gain.setValueAtTime(INITIAL_GAIN, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(
          FINAL_GAIN,
          audioContext.currentTime + duration / 1000
        );

        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + duration / 1000);

        console.log('🔊 Beep played successfully');
      } catch (error) {
        console.warn('Audio playback failed:', error);
        try {
          if (window.navigator && window.navigator.vibrate) {
            window.navigator.vibrate(FALLBACK_VIBRATION_PATTERN);
            console.log('📳 Fallback to vibration');
          }
        } catch (vibrationError) {
          console.warn('Vibration fallback also failed:', vibrationError);
        }
      }
    },
    [initializeAudioContext]
  );

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
    (text) => {
      try {
        if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel();
          window.speechSynthesis.resume();
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.lang = 'en-US';
          utterance.rate = 1.1;
          utterance.pitch = 1.1;
          utterance.volume = 1.0;
          if (preferredVoiceRef.current) utterance.voice = preferredVoiceRef.current;
          window.speechSynthesis.speak(utterance);
          console.log(`Speech: "${text}"`);
        } else {
          console.warn('Speech synthesis not available, falling back to beep');
          playBeep();
        }
      } catch (error) {
        console.warn('Speech failed, falling back to beep:', error);
        playBeep();
      }
    },
    [playBeep]
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
    previousInferenceErrorRef.current = '';
    initialTaskInfoSyncRef.current = false;

    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    setConnected(false);
    dispatch(setHeartbeatStatus('disconnected'));
    console.log('ROS task status cleanup completed');
  }, [dispatch, unsubscribeFromTopic]);

  useEffect(() => {
    const enableAudioOnUserGesture = () => {
      const audioContext = initializeAudioContext();
      if (audioContext.state === 'suspended') {
        audioContext
          .resume()
          .then(() => {
            console.log('🎵 Audio enabled by user gesture');
          })
          .catch((error) => {
            console.warn('Failed to resume AudioContext on user gesture:', error);
          });
      }
    };

    const events = ['touchstart', 'touchend', 'mousedown', 'keydown', 'click'];
    events.forEach((event) => {
      document.addEventListener(event, enableAudioOnUserGesture, { once: true, passive: true });
    });

    return () => {
      events.forEach((event) => {
        document.removeEventListener(event, enableAudioOnUserGesture);
      });
    };
  }, [initializeAudioContext]);

  const subscribeToRecordingStatus = useCallback(async () => {
    try {
      const RECORDING_BEEP_FREQUENCY = 1000;
      const RECORDING_BEEP_DURATION = 400;
      const BEEP_DELAY = 100;

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

        if (
          currentPhase === RecordPhase.RECORDING &&
          previousPhase !== RecordPhase.RECORDING
        ) {
          console.log('Recording started - playing beep sound');
          setTimeout(() => {
            playBeep(RECORDING_BEEP_FREQUENCY, RECORDING_BEEP_DURATION);
          }, BEEP_DELAY);
          toast.success('Recording started!');
        }
        previousRecordPhaseRef.current = currentPhase;

        // SAVING (post-record encode) reports progress via encoding_progress.
        // Conversion progress is on /data/status (DataOperationStatus,
        // OP_CONVERSION) routed through editDatasetSlice.conversionStatus.
        const encodingProgress =
          currentPhase === RecordPhase.SAVING ? (msg.encoding_progress || 0) : 0;

        const isRunning =
          currentPhase === RecordPhase.RECORDING ||
          currentPhase === RecordPhase.SAVING;

        // Adopt robot_type as the global value when present.
        if (msg.robot_type && msg.robot_type.trim() !== '' &&
            msg.robot_type !== store.getState().tasks.robotType) {
          dispatch(selectRobotType(msg.robot_type));
        }

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
            userId: msg.task_info?.user_id || '',
            usedStorageSize: msg.used_storage_size || 0,
            totalStorageSize: msg.total_storage_size || 0,
            usedCpu: msg.used_cpu || 0,
            usedRamSize: msg.used_ram_size || 0,
            totalRamSize: msg.total_ram_size || 0,
            topicReceived: true,
          })
        );

        // Seed taskInfo once on the first non-empty echo.
        const hasTaskInfo = msg.task_info && (
          (msg.task_info.task_name && msg.task_info.task_name.length > 0) ||
          (msg.task_info.policy_path && msg.task_info.policy_path.length > 0)
        );
        if (hasTaskInfo && !initialTaskInfoSyncRef.current) {
          initialTaskInfoSyncRef.current = true;
          dispatch(
            setTaskInfo({
              taskNum: msg.task_info.task_num || '',
              taskName: msg.task_info.task_name || '',
              taskType: msg.task_info.task_type || '',
              taskInstruction: msg.task_info.task_instruction || [],
              policyPath: msg.task_info.policy_path || '',
              recordInferenceMode: msg.task_info.record_inference_mode || false,
              userId: msg.task_info.user_id || '',
              controlHz: msg.task_info.control_hz || 100,
              inferenceHz: msg.task_info.inference_hz || 15,
              chunkAlignWindowS: msg.task_info.chunk_align_window_s || 0.3,
              includeRobotisLicense: Boolean(msg.task_info.include_robotis_license),
              warmupTime: msg.task_info.warmup_time_s || 0,
              episodeTime: msg.task_info.episode_time_s || 0,
              resetTime: msg.task_info.reset_time_s || 0,
              numEpisodes: msg.task_info.num_episodes || 0,
              pushToHub: msg.task_info.push_to_hub || false,
              privateMode: msg.task_info.private_mode || false,
              useOptimizedSave: msg.task_info.use_optimized_save_mode || false,
              recordRosBag2: msg.task_info.record_rosbag2 || false,
            })
          );
        }
      });
    } catch (error) {
      console.error('Failed to subscribe to recording status topic:', error);
    }
  }, [dispatch, rosbridgeUrl, playBeep]);

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
        if (msg.operation_type !== OP_CONVERSION) {
          // OP_HF has its own /huggingface/status feed; OP_RECORDING and
          // OP_EDIT aren't surfaced as live progress in the UI today.
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

        if (msg.robot_type && msg.robot_type.trim() !== '' &&
            msg.robot_type !== store.getState().tasks.robotType) {
          dispatch(selectRobotType(msg.robot_type));
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
