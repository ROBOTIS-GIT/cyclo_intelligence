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

import React, { useEffect, useRef } from 'react';
import clsx from 'clsx';
import { MdHome, MdVideocam, MdMemory, MdWidgets, MdAccountTree } from 'react-icons/md';
import { GoGraph } from 'react-icons/go';
import { Toaster } from 'react-hot-toast';
import toast from 'react-hot-toast';
import './App.css';
import ThemeToggle from './components/ThemeToggle';
import HomePage from './pages/HomePage';
import RecordPage from './pages/RecordPage';
import InferencePage from './pages/InferencePage';
import TrainingPage from './pages/TrainingPage';
import EditDatasetPage from './pages/EditDatasetPage';
import BTManagerPage from './pages/BTManagerPage';
import { useRosTopicSubscription } from './hooks/useRosTopicSubscription';
import rosConnectionManager from './utils/rosConnectionManager';
import { useDispatch, useSelector } from 'react-redux';
import { moveToPage, persistCurrentPage } from './features/ui/uiSlice';
import { persistRobotType } from './features/tasks/taskSlice';
import PageType from './constants/pageType';

function App() {
  const dispatch = useDispatch();
  const recordTopicReceived = useSelector(
    (state) => state.tasks.recordStatus.topicReceived
  );
  const inferenceTopicReceived = useSelector(
    (state) => state.tasks.inferenceStatus.topicReceived
  );
  const taskInfo = useSelector((state) => state.tasks.taskInfo);
  const trainingTopicReceived = useSelector((state) => state.training.topicReceived);

  const page = useSelector((state) => state.ui.currentPage);
  const restoredPageFromSession = useSelector(
    (state) => state.ui.restoredPageFromSession
  );
  const robotType = useSelector((state) => state.tasks.robotType);
  const hasSyncedTaskInfo = useSelector((state) => Boolean(
    state.tasks.taskInfoSync.serverTaskInfo ||
    state.tasks.inferenceTaskInfoSync.serverTaskInfo
  ));
  const taskStatusReceived = recordTopicReceived || inferenceTopicReceived;

  const isFirstLoad = useRef(true);

  // Subscribe to task status from ROS topic (always active)
  const rosSubscriptionControls = useRosTopicSubscription();

  // rosHost is now seeded by rosSlice initialState (window.location.hostname),
  // so we no longer need to dispatch it here.

  // Register the on-connected callback once.
  useEffect(() => {
    rosConnectionManager.setOnConnected(rosSubscriptionControls.initializeSubscriptions);
  }, [rosSubscriptionControls.initializeSubscriptions]);

  // Disconnect ROS connection when app unmounts
  useEffect(() => {
    return () => {
      console.log('App unmounting, cleaning up global ROS connection');
      rosConnectionManager.disconnect();
    };
  }, []);

  // Click anywhere on a toast to dismiss it.
  useEffect(() => {
    const handler = () => toast.dismiss();
    const attach = () => {
      const el = document.getElementById('_rht_toaster');
      if (el) {
        el.style.cursor = 'pointer';
        el.addEventListener('click', handler);
      }
    };
    // Toaster mounts after first render; retry once.
    attach();
    const t = setTimeout(attach, 500);
    return () => {
      clearTimeout(t);
      const el = document.getElementById('_rht_toaster');
      if (el) el.removeEventListener('click', handler);
    };
  }, []);

  useEffect(() => {
    persistCurrentPage(page);
  }, [page]);

  useEffect(() => {
    persistRobotType(robotType);
  }, [robotType]);

  useEffect(() => {
    if (isFirstLoad.current && restoredPageFromSession) {
      isFirstLoad.current = false;
      return;
    }

    if (
      isFirstLoad.current &&
      page === PageType.HOME &&
      taskStatusReceived &&
      hasSyncedTaskInfo
    ) {
      if (taskInfo?.taskType === PageType.RECORD) {
        dispatch(moveToPage(PageType.RECORD));
      } else if (taskInfo?.taskType === PageType.INFERENCE) {
        dispatch(moveToPage(PageType.INFERENCE));
      }
      isFirstLoad.current = false;
    } else if (isFirstLoad.current && page === PageType.HOME && trainingTopicReceived) {
      dispatch(moveToPage(PageType.TRAINING));
      isFirstLoad.current = false;
    }
  }, [
    dispatch,
    hasSyncedTaskInfo,
    page,
    restoredPageFromSession,
    taskInfo?.taskType,
    taskStatusReceived,
    trainingTopicReceived,
  ]);

  const handleHomePageNavigation = () => {
    isFirstLoad.current = false;
    dispatch(moveToPage(PageType.HOME));
  };

  const handleAppHubNavigation = () => {
    if (typeof window === 'undefined') return;
    window.location.href = 'http://localhost:3000/app';
  };

  // Check conditions for Record page navigation
  const handleRecordPageNavigation = () => {
    if (process.env.REACT_APP_DEBUG === 'true') {
      console.log('handleRecordPageNavigation');
      isFirstLoad.current = false;
      dispatch(moveToPage(PageType.RECORD));
      return;
    }

    // Allow navigation if task is in progress
    if (robotType && robotType !== '') {
      console.log('robot type:', robotType, '=> allowing navigation to Record page');
      isFirstLoad.current = false;
      dispatch(moveToPage(PageType.RECORD));
      return;
    }

    // Block navigation if robot type is not set
    if (!robotType || robotType.trim() === '') {
      toast.error('Please select a robot type first in the Home page', {
        duration: 4000,
      });
      console.log('Robot type not set, blocking navigation to Record page');
      return;
    }

    // Allow navigation if conditions are met
    console.log('Robot type set, allowing navigation to Record page');
    dispatch(moveToPage(PageType.RECORD));
  };

  const handleInferencePageNavigation = () => {
    if (process.env.REACT_APP_DEBUG === 'true') {
      console.log('handleInferencePageNavigation');
      isFirstLoad.current = false;
      dispatch(moveToPage(PageType.INFERENCE));
      return;
    }

    // Allow navigation if task is in progress
    if (robotType && robotType !== '') {
      console.log('robot type:', robotType, '=> allowing navigation to Inference page');
      isFirstLoad.current = false;
      dispatch(moveToPage(PageType.INFERENCE));
      return;
    }

    // Block navigation if robot type is not set
    if (!robotType || robotType.trim() === '') {
      toast.error('Please select a robot type first in the Home page', {
        duration: 4000,
      });
      console.log('Robot type not set, blocking navigation to Inference page');
      return;
    }

    // Allow navigation if conditions are met
    console.log('Robot type set, allowing navigation to Inference page');
    dispatch(moveToPage(PageType.INFERENCE));
  };

  const handleTrainingPageNavigation = () => {
    if (process.env.REACT_APP_DEBUG === 'true') {
      console.log('handleTrainingPageNavigation');
      isFirstLoad.current = false;
      dispatch(moveToPage(PageType.TRAINING));
      return;
    }

    // Allow navigation if task is in progress
    if (robotType && robotType !== '') {
      console.log('robot type:', robotType, '=> allowing navigation to Training Guide page');
      isFirstLoad.current = false;
      dispatch(moveToPage(PageType.TRAINING));
      return;
    }

    // Block navigation if robot type is not set
    if (!robotType || robotType.trim() === '') {
      toast.error('Please select a robot type first in the Home page', {
        duration: 4000,
      });
      console.log('Robot type not set, blocking navigation to Training Guide page');
      return;
    }

    // Allow navigation if conditions are met
    console.log('Robot type set, allowing navigation to Training Guide page');
    dispatch(moveToPage(PageType.TRAINING));
  };

  const handleEditDatasetPageNavigation = () => {
    isFirstLoad.current = false;
    dispatch(moveToPage(PageType.EDIT_DATASET));
  };

  const handleBTManagerPageNavigation = () => {
    isFirstLoad.current = false;
    dispatch(moveToPage(PageType.BT_MANAGER));
  };

  const classPageButton = clsx(
    'flex',
    'flex-col',
    'items-center',
    'rounded-2xl',
    'border-none',
    'py-5',
    'px-4',
    'text-base',
    'text-gray-800',
    'dark:text-slate-100',
    'cursor-pointer',
    'transition-colors',
    'duration-150',
    'outline-none',
    'w-24'
  );

  const classShortcutButton = clsx(
    'h-8',
    'rounded-full',
    'border',
    'border-gray-200',
    'bg-white',
    'px-2',
    'text-xs',
    'font-semibold',
    'text-gray-700',
    'shadow-sm',
    'transition-all',
    'duration-150',
    'hover:border-blue-400',
    'hover:text-blue-600',
    'hover:shadow-md',
    'dark:border-slate-700',
    'dark:bg-slate-800',
    'dark:text-slate-100',
    'dark:hover:border-blue-400',
    'dark:hover:text-blue-300'
  );

  const managerHref =
    typeof window === 'undefined'
      ? 'http://localhost:3000/home'
      : `http://${window.location.hostname}:3000/home`;

  return (
    <div className="flex min-h-screen w-screen bg-white text-gray-900 dark:bg-slate-950 dark:text-slate-100">
      <aside className="w-30 min-w-28 bg-gray-100 dark:bg-slate-900 min-h-screen flex flex-col items-center gap-4 shadow-[inset_0_0_2px_rgba(0,0,0,0.1)] dark:shadow-[inset_0_0_0_1px_rgba(148,163,184,0.12)]">
        <div className="w-full h-screen flex flex-col gap-2 items-center overflow-y-auto scrollbar-thin">
          <div className="w-full px-2 pt-3 pb-2 flex flex-col gap-2 border-b border-gray-200 dark:border-slate-800">
            <ThemeToggle />
            <div className="flex items-center justify-center gap-1.5">
              <button
                type="button"
                className={clsx(classShortcutButton, 'min-w-12', {
                  'bg-gray-300 text-gray-900 dark:bg-slate-700 dark:text-white': page === PageType.HOME,
                })}
                onClick={handleAppHubNavigation}
                title="Cyclo Apps"
                aria-label="Cyclo Apps"
              >
                Home
              </button>
              <a
                href={managerHref}
                className={clsx(classShortcutButton, 'w-8 px-0 flex items-center justify-center no-underline')}
                title="Cyclo Manager"
                aria-label="Cyclo Manager"
              >
                M
              </a>
              <button
                type="button"
                className={clsx(classShortcutButton, 'w-8 px-0', {
                  'bg-gray-300 text-gray-900 dark:bg-slate-700 dark:text-white': page === PageType.HOME,
                })}
                onClick={handleHomePageNavigation}
                title="Cyclo Intelligence"
                aria-label="Cyclo Intelligence"
              >
                C
              </button>
            </div>
          </div>
          {/* Home page button */}
          <button
            className={clsx(classPageButton, {
              'hover:bg-gray-200 active:bg-gray-400 dark:hover:bg-slate-800 dark:active:bg-slate-700': page !== PageType.HOME,
              'bg-gray-300 dark:bg-slate-700': page === PageType.HOME,
            })}
            onClick={handleHomePageNavigation}
          >
            <MdHome size={32} className="mb-1.5" />
            <span className="mt-1 text-sm">Home</span>
          </button>

          {/* Record page button */}
          <button
            className={clsx(classPageButton, {
              'hover:bg-gray-200 active:bg-gray-400 dark:hover:bg-slate-800 dark:active:bg-slate-700': page !== PageType.RECORD,
              'bg-gray-300 dark:bg-slate-700': page === PageType.RECORD,
            })}
            onClick={handleRecordPageNavigation}
          >
            <MdVideocam size={32} className="mb-1.5" />
            <span className="mt-1 text-sm">Record</span>
          </button>
          {/* Training Guide page button */}
          <button
            className={clsx(classPageButton, {
              'hover:bg-gray-200 active:bg-gray-400 dark:hover:bg-slate-800 dark:active:bg-slate-700': page !== PageType.TRAINING,
              'bg-gray-300 dark:bg-slate-700': page === PageType.TRAINING,
            })}
            onClick={handleTrainingPageNavigation}
          >
            <GoGraph size={28} className="mb-1.5" />
            <span className="mt-1 text-center text-sm leading-tight">
              Training<br />Guide
            </span>
          </button>
          {/* Inference page button */}
          <button
            className={clsx(classPageButton, {
              'hover:bg-gray-200 active:bg-gray-400 dark:hover:bg-slate-800 dark:active:bg-slate-700': page !== PageType.INFERENCE,
              'bg-gray-300 dark:bg-slate-700': page === PageType.INFERENCE,
            })}
            onClick={handleInferencePageNavigation}
          >
            <MdMemory size={32} className="mb-1.5" />
            <span className="mt-1 text-sm">Inference</span>
          </button>

          {/* BT Manager page button */}
          <button
            className={clsx(classPageButton, {
              'hover:bg-gray-200 active:bg-gray-400 dark:hover:bg-slate-800 dark:active:bg-slate-700': page !== PageType.BT_MANAGER,
              'bg-gray-300 dark:bg-slate-700': page === PageType.BT_MANAGER,
            })}
            onClick={handleBTManagerPageNavigation}
          >
            <MdAccountTree size={28} className="mb-2" />
            <span className="mt-1 text-sm whitespace-nowrap">BT Manager</span>
          </button>

          {/* Divider line */}
          <div className="w-24 h-1 border-t-2 rounded-full border-gray-200 dark:border-slate-800 mt-3"></div>

          {/* Edit dataset page button */}
          <button
            className={clsx(classPageButton, {
              'hover:bg-gray-200 active:bg-gray-400 dark:hover:bg-slate-800 dark:active:bg-slate-700': page !== PageType.EDIT_DATASET,
              'bg-gray-300 dark:bg-slate-700': page === PageType.EDIT_DATASET,
            })}
            onClick={handleEditDatasetPageNavigation}
          >
            <MdWidgets size={28} className="mb-2" />
            <span className="mt-1 text-sm whitespace-nowrap">Data Tools</span>
          </button>

        </div>
      </aside>
      <main className="flex-1 flex flex-col h-screen bg-white dark:bg-slate-950">
        {page === PageType.HOME ? (
          <HomePage />
        ) : page === PageType.RECORD ? (
          <RecordPage isActive={page === PageType.RECORD} />
        ) : page === PageType.INFERENCE ? (
          <InferencePage isActive={page === PageType.INFERENCE} />
        ) : page === PageType.TRAINING ? (
          <TrainingPage isActive={page === PageType.TRAINING} />
        ) : page === PageType.EDIT_DATASET ? (
          <EditDatasetPage isActive={page === PageType.EDIT_DATASET} />
        ) : page === PageType.BT_MANAGER ? (
          <BTManagerPage isActive={page === PageType.BT_MANAGER} />
        ) : (
          <HomePage />
        )}
      </main>
      <Toaster
        position="top-center"
        gutter={8}
        toastOptions={{
          duration: 3000,
          style: {
            background: '#363636',
            color: '#fff',
            maxWidth: '500px',
            wordWrap: 'break-word',
            whiteSpace: 'pre-wrap',
            lineHeight: '1.4',
          },
          success: {
            duration: 3000,
            style: {
              background: '#10b981',
              maxWidth: '500px',
              wordWrap: 'break-word',
              whiteSpace: 'pre-wrap',
              lineHeight: '1.4',
            },
          },
          error: {
            duration: 6000,
            style: {
              background: '#ef4444',
              maxWidth: '500px',
              wordWrap: 'break-word',
              whiteSpace: 'pre-wrap',
              lineHeight: '1.4',
            },
          },
        }}
      />
    </div>
  );
}

export default App;
