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

import React, { useState, useCallback, useEffect, useMemo } from 'react';
import clsx from 'clsx';
import { useSelector, useDispatch } from 'react-redux';
import toast from 'react-hot-toast';
import { MdFolderOpen, MdOutlineFileUpload, MdOutlineFileDownload } from 'react-icons/md';
import { SiHuggingface } from 'react-icons/si';
import {
  setHFActiveEndpoint,
  setHFEndpoints,
  setHFUserId,
  setHFRepoIdUpload,
  setHFRepoIdDownload,
} from '../editDatasetSlice';
import { useRosServiceCaller } from '../../../hooks/useRosServiceCaller';
import FileBrowserModal from '../../../components/FileBrowserModal';
import TokenInputPopup from '../../../components/TokenInputPopup';
import SectionSelector from './SectionSelector';
import { DEFAULT_PATHS, HF_ENDPOINT_PRESETS } from '../../../constants/paths';
import HFStatus from '../../../constants/HFStatus';

// Sentinel value used by the endpoint dropdown to mean "let the user type a
// custom URL". Anything else in the dropdown is a real endpoint URL.
const CUSTOM_ENDPOINT_SENTINEL = '__custom__';

// Constants
const SECTION_NAME = {
  UPLOAD: 'upload',
  DOWNLOAD: 'download',
};

// HuggingFace repository name validation
const validateHfRepoName = (repoName) => {
  if (!repoName) return { isValid: false, message: '' };

  // Check length (max 96 characters)
  if (repoName.length > 96) {
    return {
      isValid: false,
      message: 'Repository name must be 96 characters or less',
    };
  }

  // Check if starts or ends with '-' or '.'
  if (
    repoName.startsWith('-') ||
    repoName.startsWith('.') ||
    repoName.endsWith('-') ||
    repoName.endsWith('.')
  ) {
    return {
      isValid: false,
      message: 'Repository name cannot start or end with "-" or "."',
    };
  }

  // Check for forbidden patterns '--' and '..'
  if (repoName.includes('--') || repoName.includes('..')) {
    return {
      isValid: false,
      message: 'Repository name cannot contain "--" or ".."',
    };
  }

  // Check for allowed characters only (alphanumeric, '-', '_', '.')
  const allowedPattern = /^[a-zA-Z0-9._-]+$/;
  if (!allowedPattern.test(repoName)) {
    return {
      isValid: false,
      message: 'Repository name can only contain letters, numbers, "-", "_", and "."',
    };
  }

  return { isValid: true, message: '' };
};

// Style Classes
const STYLES = {
  textInput: clsx(
    'text-sm',
    'w-full',
    'h-10',
    'p-2',
    'border',
    'border-gray-300',
    'rounded-md',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-blue-500',
    'focus:border-transparent'
  ),
  selectUserID: clsx(
    'text-md',
    'w-full',
    'max-w-120',
    'min-w-60',
    'h-8',
    'px-2',
    'border',
    'border-gray-300',
    'rounded-md',
    'focus:outline-none',
    'focus:ring-2',
    'focus:ring-blue-500',
    'focus:border-transparent'
  ),
  loadUserButton: clsx('px-3', 'py-1', 'text-md', 'font-medium', 'rounded-xl', 'transition-colors'),
  cancelButton: clsx('px-6', 'py-2', 'text-sm', 'font-medium', 'rounded-lg', 'transition-colors'),
};

// Folder Browse Button Component
const FolderBrowseButton = ({ onClick, disabled = false, ariaLabel }) => {
  return (
    <button
      type="button"
      onClick={onClick}
      className={clsx('flex items-center justify-center w-10 h-10 rounded-md transition-colors', {
        'text-blue-500 bg-gray-200 hover:text-blue-700': !disabled,
        'text-gray-400 bg-gray-100 cursor-not-allowed': disabled,
      })}
      aria-label={ariaLabel}
      disabled={disabled}
    >
      <MdFolderOpen className="w-8 h-8" />
    </button>
  );
};

const HuggingfaceSection = () => {
  const dispatch = useDispatch();
  const userId = useSelector((state) => state.editDataset.hfUserId);
  const hfRepoIdUpload = useSelector((state) => state.editDataset.hfRepoIdUpload);
  const hfRepoIdDownload = useSelector((state) => state.editDataset.hfRepoIdDownload);
  const hfStatus = useSelector((state) => state.editDataset.hfStatus);
  const downloadStatus = useSelector((state) => state.editDataset.downloadStatus);
  const uploadStatus = useSelector((state) => state.editDataset.uploadStatus);
  const hfActiveEndpoint = useSelector((state) => state.editDataset.hfActiveEndpoint);
  const hfEndpoints = useSelector((state) => state.editDataset.hfEndpoints);

  const {
    controlHfServer,
    registerHFUser,
    getRegisteredHFUser,
    listHFEndpoints,
    selectHFEndpoint,
  } = useRosServiceCaller();

  // Local states
  const [activeSection, setActiveSection] = useState(SECTION_NAME.UPLOAD);
  const [hfLocalDirUpload, setHfLocalDirUpload] = useState('');
  // Per-section repo-type toggles. Both sections accept either a dataset
  // or a model; the toggle just changes the HF API repo_type on the wire
  // and the default destination path for download.
  const [uploadType, setUploadType] = useState('dataset');
  const [downloadType, setDownloadType] = useState('model');
  // Destination directory for HF downloads. Default tracks the selected
  // download type (model -> /workspace/model, dataset -> /workspace/rosbag2);
  // the user can override via the input + browser.
  const [hfLocalDirDownload, setHfLocalDirDownload] = useState(
    DEFAULT_PATHS.HF_MODEL_DOWNLOAD_PATH
  );
  // When the user toggles type, swap the destination to that type's
  // canonical default — but only if the current value is a known default
  // (i.e. they haven't typed something custom). Custom values stay put.
  useEffect(() => {
    const knownDefaults = new Set([
      DEFAULT_PATHS.HF_MODEL_DOWNLOAD_PATH,
      DEFAULT_PATHS.HF_DATASET_DOWNLOAD_PATH,
    ]);
    if (knownDefaults.has(hfLocalDirDownload)) {
      setHfLocalDirDownload(
        downloadType === 'model'
          ? DEFAULT_PATHS.HF_MODEL_DOWNLOAD_PATH
          : DEFAULT_PATHS.HF_DATASET_DOWNLOAD_PATH
      );
    }
    // hfLocalDirDownload intentionally omitted — we only react to type flips,
    // not to the user editing the field.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [downloadType]);

  const [showHfDownloadDirBrowserModal, setShowHfDownloadDirBrowserModal] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [showHfLocalDirBrowserModal, setShowHfLocalDirBrowserModal] = useState(false);
  const [userIdList, setUserIdList] = useState([]);
  const [showTokenPopup, setShowTokenPopup] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Validation states
  const [uploadRepoValidation, setUploadRepoValidation] = useState({ isValid: true, message: '' });
  const [downloadRepoValidation, setDownloadRepoValidation] = useState({
    isValid: true,
    message: '',
  });

  // Computed values
  const isProcessing = isUploading || isDownloading;

  // Section availability
  const canChangeSection = !isProcessing;

  const isHfStatusReady =
    hfStatus === HFStatus.IDLE || hfStatus === HFStatus.SUCCESS || hfStatus === HFStatus.FAILED;

  const uploadButtonEnabled =
    !isUploading &&
    !isDownloading &&
    hfRepoIdUpload?.trim() &&
    hfLocalDirUpload?.trim() &&
    uploadRepoValidation.isValid &&
    userId?.trim() &&
    isHfStatusReady;

  const downloadButtonEnabled =
    !isUploading &&
    !isDownloading &&
    hfRepoIdDownload?.trim() &&
    downloadRepoValidation.isValid &&
    userId?.trim() &&
    isHfStatusReady;
  // Button variants helper function
  const getButtonVariant = (variant, isActive = true, isLoading = false) => {
    const variants = {
      blue: {
        active: 'bg-blue-200 text-blue-800 hover:bg-blue-300',
        disabled: 'bg-gray-200 text-gray-500 cursor-not-allowed',
      },
      red: {
        active: 'bg-red-200 text-red-800 hover:bg-red-300',
        disabled: 'bg-gray-200 text-gray-500 cursor-not-allowed',
      },
      green: {
        active: 'bg-green-200 text-green-800 hover:bg-green-300',
        disabled: 'bg-gray-200 text-gray-500 cursor-not-allowed',
      },
    };

    const isDisabled = !isActive || isLoading;
    return variants[variant]?.[isDisabled ? 'disabled' : 'active'] || '';
  };

  // Refresh the in-memory list of registered endpoints from the server. The
  // active endpoint and the user list of that endpoint are populated as a
  // side effect so the rest of the section stays in sync.
  const refreshEndpoints = useCallback(
    async ({ silent = false } = {}) => {
      try {
        const result = await listHFEndpoints();
        if (!result?.success) {
          if (!silent) toast.error(result?.message || 'Failed to list HF endpoints');
          return null;
        }
        const endpoints = (result.endpoints || []).map((ep, idx) => ({
          endpoint: ep,
          label: result.labels?.[idx] || '',
          userId: result.user_ids?.[idx] || '',
        }));
        dispatch(setHFEndpoints(endpoints));
        dispatch(setHFActiveEndpoint(result.active || ''));
        // Default the displayed userId to whatever the server has on file
        // for the active endpoint, so the upload/download forms have a
        // sensible username right after a refresh.
        const activeEntry = endpoints.find((e) => e.endpoint === result.active);
        if (activeEntry?.userId) {
          dispatch(setHFUserId(activeEntry.userId));
        }
        return result;
      } catch (error) {
        console.warn('Error listing HF endpoints:', error);
        if (!silent) toast.error(`Failed to list HF endpoints: ${error.message}`);
        return null;
      }
    },
    [listHFEndpoints, dispatch]
  );

  // Submit a token for the *currently active* endpoint. The popup is opened
  // from the endpoint row, so by the time we get here `hfActiveEndpoint`
  // already names the endpoint we want to register the token against.
  const handleTokenSubmit = async ({ token, label = '' } = {}) => {
    const trimmed = (token || '').trim();
    if (!trimmed) {
      toast.error('Please enter a token');
      return;
    }
    if (!hfActiveEndpoint) {
      toast.error('Pick or type an endpoint URL first');
      return;
    }

    setIsLoading(true);
    try {
      const result = await registerHFUser({
        endpoint: hfActiveEndpoint,
        label,
        token: trimmed,
      });
      console.log('registerHFUser result:', result);

      if (result?.success && result.user_id_list) {
        setUserIdList(result.user_id_list);
        if (result.user_id_list.length > 0) {
          dispatch(setHFUserId(result.user_id_list[0]));
        }
        setShowTokenPopup(false);
        toast.success(
          `Token registered for ${hfActiveEndpoint} (${result.user_id_list[0] || 'unknown'})`
        );
        await refreshEndpoints({ silent: true });
      } else {
        toast.error(result?.message || 'Failed to register token');
      }
    } catch (error) {
      console.error('Error registering HF user:', error);
      toast.error(`Failed to register user: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  // Pull the user list for ``targetEndpoint`` (or the currently active one).
  // We pass the endpoint explicitly so callers that have *just* changed the
  // active endpoint don't accidentally read a stale closure value of
  // ``hfActiveEndpoint`` (Redux dispatches are not synchronous from the
  // perspective of the same render).
  const loadUserIdsForEndpoint = useCallback(
    async (targetEndpoint, { silent = false } = {}) => {
      const ep = (targetEndpoint || '').trim();
      if (!ep) {
        if (!silent) toast.error('No HuggingFace endpoint selected');
        return;
      }
      setIsLoading(true);
      try {
        const result = await getRegisteredHFUser(ep);
        console.log('getRegisteredHFUser result:', result);

        if (result && result.user_id_list) {
          if (result.success) {
            setUserIdList(result.user_id_list);
            if (result.user_id_list.length > 0) {
              dispatch(setHFUserId(result.user_id_list[0]));
            } else {
              dispatch(setHFUserId(''));
            }
            if (!silent) toast.success(`Loaded user list for ${ep}`);
          } else {
            // Server-side failure for this specific endpoint — keep the UI
            // honest by clearing the user list rather than showing the
            // previous endpoint's users.
            setUserIdList([]);
            dispatch(setHFUserId(''));
            if (!silent) {
              toast.error('Failed to get user ID list:\n' + result.message);
            }
          }
        } else if (!silent) {
          toast.error('Failed to get user ID list from response');
        }
      } catch (error) {
        console.warn('Error loading HF user list:', error);
        if (!silent) {
          toast.error(`Failed to load user ID list: ${error.message}`);
        }
      } finally {
        setIsLoading(false);
      }
    },
    [getRegisteredHFUser, dispatch]
  );

  // Convenience wrapper used by the explicit "Load" button — always operates
  // on the currently active endpoint.
  const handleLoadUserId = useCallback(
    ({ silent = false } = {}) => loadUserIdsForEndpoint(hfActiveEndpoint, { silent }),
    [loadUserIdsForEndpoint, hfActiveEndpoint]
  );

  // Switch the active endpoint server-side, then refresh the local cache and
  // pull the user list for the *new* endpoint so the User ID dropdown reflects
  // reality.
  const handleEndpointChange = useCallback(
    async (newEndpoint) => {
      if (!newEndpoint) return;
      try {
        const isRegistered = hfEndpoints.some((e) => e.endpoint === newEndpoint);
        if (isRegistered) {
          const result = await selectHFEndpoint(newEndpoint);
          if (!result?.success) {
            toast.error(result?.message || 'Failed to select endpoint');
            return;
          }
          dispatch(setHFActiveEndpoint(newEndpoint));
          await refreshEndpoints({ silent: true });
          // Crucial: refetch the user list for the newly selected endpoint.
          // Without this, the User ID dropdown would keep showing whatever
          // users belonged to the previously active endpoint.
          await loadUserIdsForEndpoint(newEndpoint, { silent: true });
        } else {
          // Not yet registered: remember it locally so the user can open the
          // token popup against this URL, and clear any leftover user list
          // from the previous endpoint to avoid confusion.
          dispatch(setHFActiveEndpoint(newEndpoint));
          setUserIdList([]);
          dispatch(setHFUserId(''));
        }
      } catch (error) {
        console.error('Error switching HF endpoint:', error);
        toast.error(`Failed to switch endpoint: ${error.message}`);
      }
    },
    [hfEndpoints, selectHFEndpoint, dispatch, refreshEndpoints, loadUserIdsForEndpoint]
  );

  // File browser handlers
  const handleHfLocalDirSelect = useCallback((item) => {
    setHfLocalDirUpload(item.full_path);
  }, []);

  // Build the full ``<owner>/<name>`` repo path from a user-provided value.
  // If the user already typed an explicit ``owner/name`` (e.g. an organization
  // they have access to), we honour it as-is and skip the userId prefix.
  // Otherwise we fall back to the currently selected userId.
  const resolveFullRepoId = (value) => {
    const trimmed = (value || '').trim();
    if (!trimmed) return '';
    if (trimmed.includes('/')) return trimmed;
    return `${userId || ''}/${trimmed}`;
  };

  // Validate just the repo name portion (after the slash) since `validateHfRepoName`
  // does not allow slashes.
  const validateRepoIdInput = (value) => {
    const trimmed = (value || '').trim();
    if (!trimmed) return validateHfRepoName('');
    const namePart = trimmed.includes('/') ? trimmed.split('/').slice(-1)[0] : trimmed;
    return validateHfRepoName(namePart);
  };

  // Input handlers with validation
  const handleUploadRepoIdChange = (value) => {
    dispatch(setHFRepoIdUpload(value));
    setUploadRepoValidation(validateRepoIdInput(value));
  };

  const handleDownloadRepoIdChange = (value) => {
    // If the input matches a registered userId followed by a slash, switch the
    // active userId to that one and store the bare repo name. Otherwise we
    // keep the raw value (including any owner/name with an organization owner)
    // and let resolveFullRepoId handle it at submit time.
    if (value.includes('/')) {
      const [head, tail] = value.split('/', 2);
      if (userIdList.includes(head)) {
        dispatch(setHFUserId(head));
        dispatch(setHFRepoIdDownload(tail));
        setDownloadRepoValidation(validateRepoIdInput(tail));
        return;
      }
    }

    dispatch(setHFRepoIdDownload(value));
    setDownloadRepoValidation(validateRepoIdInput(value));
  };

  // Operations
  const operations = {
    uploadDataset: async () => {
      if (!hfRepoIdUpload || hfRepoIdUpload.trim() === '') {
        toast.error('Please enter a Repo ID first');
        return;
      }

      if (!hfLocalDirUpload || hfLocalDirUpload.trim() === '') {
        toast.error('Please select a Local Directory first');
        return;
      }

      // Additional validation check (validates the bare repo name portion)
      const validation = validateRepoIdInput(hfRepoIdUpload);
      if (!validation.isValid) {
        toast.error(`Invalid repository name: ${validation.message}`);
        return;
      }

      setIsUploading(true);
      try {
        const repoId = resolveFullRepoId(hfRepoIdUpload);
        if (!repoId.includes('/')) {
          toast.error(
            'No HuggingFace user ID is selected. Either pick one from the dropdown ' +
              'or type the full owner/name (e.g. my-org/my-dataset).'
          );
          setIsUploading(false);
          return;
        }
        const localDir = hfLocalDirUpload.trim();
        const result = await controlHfServer(
          'upload',
          repoId,
          uploadType,
          localDir,
          hfActiveEndpoint
        );
        console.log(`Upload ${uploadType} result:`, result);
        toast.success(`Upload started! (${repoId})`);
      } catch (error) {
        console.error(`Error uploading ${uploadType}:`, error);
        toast.error(`Failed to upload ${uploadType}: ${error.message}`);
      } finally {
        setIsUploading(false);
      }
    },

    downloadDataset: async () => {
      if (!hfRepoIdDownload || hfRepoIdDownload.trim() === '') {
        toast.error('Please enter a Repo ID first');
        return;
      }

      // Additional validation check (validates the bare repo name portion)
      const validation = validateRepoIdInput(hfRepoIdDownload);
      if (!validation.isValid) {
        toast.error(`Invalid repository name: ${validation.message}`);
        return;
      }

      setIsDownloading(true);
      try {
        const repoId = resolveFullRepoId(hfRepoIdDownload);
        if (!repoId.includes('/')) {
          toast.error(
            'No HuggingFace user ID is selected. Either pick one from the dropdown ' +
              'or type the full owner/name (e.g. my-org/my-dataset).'
          );
          setIsDownloading(false);
          return;
        }
        const result = await controlHfServer(
          'download',
          repoId,
          downloadType,
          (hfLocalDirDownload || '').trim(),
          hfActiveEndpoint
        );
        console.log(`Download ${downloadType} result:`, result);

        toast.success(`Download started!\n(${repoId})`);
      } catch (error) {
        console.error(`Error downloading ${downloadType}:`, error);
        toast.error(`Failed to download ${downloadType}: ${error.message}`);
      } finally {
        setIsDownloading(false);
      }
    },
    cancelOperation: async () => {
      try {
        // Cancel applies to whichever transfer is in progress; the type is
        // unused server-side for cancellations but we still pass a sane value.
        const result = await controlHfServer(
          'cancel',
          hfRepoIdDownload,
          'model',
          '',
          hfActiveEndpoint
        );
        console.log('Cancel download result:', result);
        toast.success(`Cancelling... (${hfRepoIdDownload})`);
      } catch (error) {
        console.error('Error canceling download:', error);
        toast.error(`Failed to cancel download: ${error.message}`);
      }
    },
  };

  // Auto-load endpoint list (and the active endpoint's user list) on mount.
  // Both calls are silent — a fresh container has nothing registered yet, and
  // we don't want to spam toasts in that perfectly normal state.
  useEffect(() => {
    refreshEndpoints({ silent: true }).then((result) => {
      if (result?.active) {
        // Pass the freshly resolved endpoint explicitly — relying on the
        // closure-captured ``hfActiveEndpoint`` here would race with the
        // pending Redux dispatch from refreshEndpoints.
        loadUserIdsForEndpoint(result.active, { silent: true });
      }
    });
    // We intentionally only run this once on mount; subsequent refreshes are
    // triggered by user actions (token submit, endpoint switch).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Build the dropdown options: presets ∪ already-registered endpoints, with
  // duplicates removed and a Custom… entry pinned to the bottom.
  const endpointDropdownOptions = useMemo(() => {
    const seen = new Map();
    HF_ENDPOINT_PRESETS.forEach((p) => {
      seen.set(p.url, { url: p.url, label: p.label, registered: false });
    });
    (hfEndpoints || []).forEach((e) => {
      seen.set(e.endpoint, {
        url: e.endpoint,
        label: e.label || seen.get(e.endpoint)?.label || '',
        registered: true,
      });
    });
    // If the active endpoint is something the user typed but hasn't yet
    // registered, make sure it still shows up so the dropdown reflects
    // reality.
    if (hfActiveEndpoint && !seen.has(hfActiveEndpoint)) {
      seen.set(hfActiveEndpoint, {
        url: hfActiveEndpoint,
        label: '(unregistered)',
        registered: false,
      });
    }
    return Array.from(seen.values());
  }, [hfEndpoints, hfActiveEndpoint]);

  // track hf status update
  useEffect(() => {
    if (hfStatus === HFStatus.UPLOADING) {
      setActiveSection(SECTION_NAME.UPLOAD);
      setIsUploading(true);
    } else if (hfStatus === HFStatus.DOWNLOADING) {
      setActiveSection(SECTION_NAME.DOWNLOAD);
      setIsDownloading(true);
    } else {
      setIsUploading(false);
      setIsDownloading(false);
    }
  }, [hfStatus]);

  return (
    <div className="w-full flex flex-col items-start justify-start bg-gray-100 p-10 gap-4 rounded-xl">
      <div className="w-full flex items-center justify-start gap-2">
        <SiHuggingface className="w-7 h-7 text-yellow-400" />
        <span className="text-2xl font-bold">Hugging Face Upload & Download</span>
      </div>

      <div className="w-full flex flex-row items-start justify-start gap-4">
        <div className="flex flex-col items-center justify-start gap-12">
          {/* User ID Selection */}
          <div className="bg-white p-5 rounded-md flex flex-col items-start justify-center gap-4 shadow-md">
            <div className="w-full flex items-center justify-start">
              <span className="text-lg font-bold">HuggingFace endpoint</span>
            </div>

            {/* Endpoint dropdown — presets + registered endpoints + Custom… */}
            <div
              className={clsx('w-full flex flex-col gap-2', {
                'opacity-50': isDownloading || isUploading,
              })}
            >
              <select
                className={STYLES.selectUserID}
                value={
                  hfActiveEndpoint &&
                  endpointDropdownOptions.some((o) => o.url === hfActiveEndpoint)
                    ? hfActiveEndpoint
                    : (hfActiveEndpoint ? hfActiveEndpoint : '')
                }
                onChange={(e) => {
                  const v = e.target.value;
                  if (v === CUSTOM_ENDPOINT_SENTINEL) {
                    // Empty default — pre-filling 'https://' silently
                    // forced HTTPS even when users meant to point at an
                    // HTTP-only self-hosted hub, which surfaced as an
                    // 'SSL: WRONG_VERSION_NUMBER' error during whoami.
                    const url = (window.prompt(
                      'Enter the full HuggingFace endpoint URL with scheme\n(e.g. https://huggingface.co)',
                      hfActiveEndpoint || ''
                    ) || '').trim();
                    if (url) handleEndpointChange(url);
                    return;
                  }
                  handleEndpointChange(v);
                }}
                disabled={isDownloading || isUploading}
              >
                <option value="" disabled>
                  Select endpoint
                </option>
                {endpointDropdownOptions.map((opt) => (
                  <option key={opt.url} value={opt.url}>
                    {opt.label ? `${opt.label} — ${opt.url}` : opt.url}
                    {opt.registered ? '  ✓' : ''}
                  </option>
                ))}
                <option value={CUSTOM_ENDPOINT_SENTINEL}>
                  Custom URL…
                </option>
              </select>
              <div className="text-xs text-gray-500">
                Active:{' '}
                <span className="font-mono text-blue-700">
                  {hfActiveEndpoint || '<none>'}
                </span>
                {hfActiveEndpoint && (
                  <>
                    {' · '}
                    {hfEndpoints.some((e) => e.endpoint === hfActiveEndpoint)
                      ? 'token registered'
                      : 'no token yet'}
                  </>
                )}
              </div>

              {/* Token register/update button — owned by the endpoint card,
                  not the User ID row, because a token belongs to one
                  specific endpoint. */}
              <button
                className={clsx(
                  STYLES.loadUserButton,
                  getButtonVariant('green', !!hfActiveEndpoint, isLoading)
                )}
                onClick={() => {
                  if (!isLoading && hfActiveEndpoint) {
                    setShowTokenPopup(true);
                  }
                }}
                disabled={isLoading || !hfActiveEndpoint}
                title={
                  hfActiveEndpoint
                    ? 'Validate a token against this endpoint and store it on the robot'
                    : 'Pick or enter an endpoint URL first'
                }
              >
                {hfEndpoints.some((e) => e.endpoint === hfActiveEndpoint)
                  ? 'Update token'
                  : 'Register token'}
              </button>
            </div>

            <div className="w-full border-t border-gray-200" />

            <div className="w-full flex items-center justify-start">
              <span className="text-lg font-bold">User ID</span>
            </div>
            <div
              className={clsx('w-full flex flex-row gap-3', {
                'opacity-50': isDownloading || isUploading,
              })}
            >
              <select
                className={STYLES.selectUserID}
                value={userId || ''}
                onChange={(e) => dispatch(setHFUserId(e.target.value))}
                disabled={isDownloading || isUploading}
              >
                <option value="">Select User ID</option>
                {userIdList.map((userId) => (
                  <option key={userId} value={userId}>
                    {userId}
                  </option>
                ))}
              </select>
              <button
                className={clsx(STYLES.loadUserButton, getButtonVariant('blue', true, isLoading))}
                onClick={() => {
                  if (!isLoading) {
                    handleLoadUserId();
                  }
                }}
                disabled={isLoading}
                title="Re-fetch the user list for the active endpoint"
              >
                {isLoading ? 'Loading...' : 'Load'}
              </button>
            </div>
          </div>

          {/* Section Selector */}
          <div className="flex items-center justify-start">
            <SectionSelector
              activeSection={activeSection}
              onSectionChange={setActiveSection}
              canChangeSection={canChangeSection}
            />
          </div>
        </div>

        {/* Active Section Content */}
        <div className="w-full">
          {activeSection === SECTION_NAME.UPLOAD && (
            <div className="w-full bg-white p-5 rounded-md flex flex-col items-start justify-center gap-2 shadow-md">
              {/* Upload Section Header */}
              <div className="w-full flex flex-col items-start justify-start gap-2 bg-gray-50 border border-gray-200 p-3 rounded-md">
                <div className="w-full flex items-center justify-between gap-2">
                  <div className="flex items-center rounded-md font-medium gap-2">
                    <MdOutlineFileUpload className="text-lg text-green-600" />
                    Upload {uploadType === 'model' ? 'Model' : 'Dataset'}
                  </div>
                  <div
                    className="flex gap-1 rounded-md bg-gray-200 p-0.5"
                    role="radiogroup"
                    aria-label="Upload repo type"
                  >
                    {['dataset', 'model'].map((t) => (
                      <button
                        key={t}
                        type="button"
                        role="radio"
                        aria-checked={uploadType === t}
                        onClick={() => setUploadType(t)}
                        disabled={isUploading}
                        className={clsx(
                          'px-3 py-0.5 text-xs font-medium rounded transition-colors',
                          uploadType === t
                            ? 'bg-white text-gray-900 shadow-sm'
                            : 'text-gray-500 hover:text-gray-700',
                          isUploading && 'cursor-not-allowed opacity-60'
                        )}
                      >
                        {t === 'dataset' ? 'Dataset' : 'Model'}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="text-sm text-gray-600">
                  <div className="mb-1">
                    Uploads a local {uploadType} folder to Hugging Face hub.
                  </div>
                </div>
              </div>

              {/* Upload Dataset Section Content */}
              <div className="w-full flex flex-col gap-3">
                {/* Local Directory Input */}
                <div className="w-full flex flex-col gap-2">
                  <span className="text-lg font-bold">Local Directory</span>
                  <div className="w-full flex flex-row items-center justify-start gap-2">
                    <FolderBrowseButton
                      onClick={() => setShowHfLocalDirBrowserModal(true)}
                      disabled={isDownloading}
                      ariaLabel="Browse files for local directory"
                    />
                    <input
                      className={clsx(STYLES.textInput, 'flex-1', {
                        'bg-gray-100 cursor-not-allowed': isDownloading,
                        'bg-white': !isDownloading,
                      })}
                      type="text"
                      placeholder="Enter local directory path or browse"
                      value={hfLocalDirUpload || ''}
                      onChange={(e) => setHfLocalDirUpload(e.target.value)}
                      disabled={isDownloading}
                    />
                  </div>
                </div>

                {/* Repo ID Input */}
                <div className="w-full flex flex-col gap-2">
                  <span className="text-lg font-bold">Repository ID</span>
                  <div className="relative">
                    <div
                      className={clsx(
                        'flex items-center border rounded-md overflow-hidden bg-white focus-within:ring-2',
                        {
                          'border-gray-300 focus-within:ring-blue-500 focus-within:border-transparent':
                            uploadRepoValidation.isValid || !hfRepoIdUpload,
                          'border-red-300 focus-within:ring-red-500 focus-within:border-transparent':
                            !uploadRepoValidation.isValid && hfRepoIdUpload,
                        }
                      )}
                    >
                      <input
                        className={clsx(
                          'flex-1 px-3 py-2 text-sm bg-transparent border-none outline-none',
                          {
                            'bg-gray-100 cursor-not-allowed text-gray-500': isUploading,
                            'text-gray-900': !isUploading,
                          }
                        )}
                        type="text"
                        placeholder="repo-name  or  org-or-user/repo-name"
                        value={hfRepoIdUpload || ''}
                        onChange={(e) => handleUploadRepoIdChange(e.target.value)}
                        disabled={isUploading}
                      />
                    </div>
                    <div className="mt-1 text-xs">
                      <div className="text-gray-500">
                        Full repository path:{' '}
                        <span className="font-mono text-blue-600">
                          {resolveFullRepoId(hfRepoIdUpload) || '—'}
                        </span>
                        {!hfRepoIdUpload?.includes('/') && (
                          <span className="text-gray-400">
                            {' '}(prefixed with selected user ID)
                          </span>
                        )}
                      </div>
                      {!uploadRepoValidation.isValid && hfRepoIdUpload && (
                        <div className="text-red-500 mt-1">⚠️ {uploadRepoValidation.message}</div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Upload Button */}
                <div className="w-full flex flex-row items-center justify-start gap-3 mt-2">
                  <button
                    className={clsx(
                      'px-6',
                      'py-2',
                      'text-sm',
                      'font-medium',
                      'rounded-lg',
                      'transition-colors',
                      {
                        'bg-green-500 text-white hover:bg-green-600': uploadButtonEnabled,
                        'bg-gray-300 text-gray-500 cursor-not-allowed': !uploadButtonEnabled,
                      }
                    )}
                    onClick={operations.uploadDataset}
                    disabled={!uploadButtonEnabled}
                  >
                    <div className="flex items-center justify-center gap-2">
                      <MdOutlineFileUpload className="w-6 h-6" />
                      Upload
                    </div>
                  </button>

                  {/* Cancel Button */}
                  <button
                    className={clsx(STYLES.cancelButton, {
                      'bg-red-500 text-white hover:bg-red-600': isUploading,
                      'bg-gray-300 text-gray-500 cursor-not-allowed': !isUploading,
                    })}
                    onClick={operations.cancelOperation}
                    disabled={!isUploading}
                  >
                    Cancel
                  </button>

                  {/* Status */}
                  <div className="flex flex-row items-center justify-start">
                    <span className="text-sm text-gray-500">
                      {isUploading && '⏳ Uploading...'}
                      {!isUploading && hfStatus}
                    </span>
                  </div>

                  {/* Upload Progress Bar */}
                  {isUploading && (
                    <div className="w-full">
                      <div className="flex flex-row items-center justify-between mb-1">
                        <span className="text-sm text-gray-500">
                          {uploadStatus.current}/{uploadStatus.total}
                        </span>
                        <span className="text-sm text-gray-500">{uploadStatus.percentage}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-out"
                          style={{ width: `${uploadStatus.percentage}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeSection === SECTION_NAME.DOWNLOAD && (
            <div className="w-full bg-white p-5 rounded-md flex flex-col items-start justify-center gap-4 shadow-md">
              {/* Download Section Header */}
              <div className="w-full flex flex-col items-start justify-start gap-2 bg-gray-50 border border-gray-200 p-3 rounded-md">
                <div className="w-full flex items-center justify-between gap-2">
                  <div className="flex items-center rounded-md font-medium gap-2">
                    <MdOutlineFileDownload className="text-lg text-blue-600" />
                    Download {downloadType === 'dataset' ? 'Dataset' : 'Model'}
                  </div>
                  <div
                    className="flex gap-1 rounded-md bg-gray-200 p-0.5"
                    role="radiogroup"
                    aria-label="Download repo type"
                  >
                    {['dataset', 'model'].map((t) => (
                      <button
                        key={t}
                        type="button"
                        role="radio"
                        aria-checked={downloadType === t}
                        onClick={() => setDownloadType(t)}
                        disabled={isDownloading}
                        className={clsx(
                          'px-3 py-0.5 text-xs font-medium rounded transition-colors',
                          downloadType === t
                            ? 'bg-white text-gray-900 shadow-sm'
                            : 'text-gray-500 hover:text-gray-700',
                          isDownloading && 'cursor-not-allowed opacity-60'
                        )}
                      >
                        {t === 'dataset' ? 'Dataset' : 'Model'}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="text-sm text-gray-600">
                  <div className="mb-1">
                    Downloads a {downloadType} from Hugging Face hub to a local directory.
                  </div>
                </div>
              </div>

              {/* Download Dataset Section Content */}
              <div className="w-full flex flex-col gap-3">
                {/* Repo ID Input */}
                <div className="w-full flex flex-col gap-2">
                  <span className="text-lg font-bold">Repository ID</span>
                  <div className="relative">
                    <div
                      className={clsx(
                        'flex items-center border rounded-md overflow-hidden bg-white focus-within:ring-2',
                        {
                          'border-gray-300 focus-within:ring-blue-500 focus-within:border-transparent':
                            downloadRepoValidation.isValid || !hfRepoIdDownload,
                          'border-red-300 focus-within:ring-red-500 focus-within:border-transparent':
                            !downloadRepoValidation.isValid && hfRepoIdDownload,
                        }
                      )}
                    >
                      <input
                        className={clsx(
                          'flex-1 px-3 py-2 text-sm bg-transparent border-none outline-none',
                          {
                            'bg-gray-100 cursor-not-allowed text-gray-500': isDownloading,
                            'text-gray-900': !isDownloading,
                          }
                        )}
                        type="text"
                        placeholder="repo-name  or  org-or-user/repo-name"
                        value={hfRepoIdDownload || ''}
                        onChange={(e) => handleDownloadRepoIdChange(e.target.value)}
                        disabled={isDownloading}
                      />
                    </div>
                    <div className="mt-1 text-xs">
                      <div className="text-gray-500">
                        Full repository path:{' '}
                        <span className="font-mono text-blue-600">
                          {resolveFullRepoId(hfRepoIdDownload) || '—'}
                        </span>
                        {!hfRepoIdDownload?.includes('/') && (
                          <span className="text-gray-400">
                            {' '}(prefixed with selected user ID)
                          </span>
                        )}
                      </div>
                      {!downloadRepoValidation.isValid && hfRepoIdDownload && (
                        <div className="text-red-500 mt-1">⚠️ {downloadRepoValidation.message}</div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Save destination — user-editable */}
                <div className="w-full flex flex-col gap-2">
                  <span className="text-lg font-bold">Save to</span>
                  <div className="w-full flex flex-row items-center justify-start gap-2">
                    <FolderBrowseButton
                      onClick={() => setShowHfDownloadDirBrowserModal(true)}
                      disabled={isDownloading}
                      ariaLabel="Browse files for download destination"
                    />
                    <input
                      className={clsx(STYLES.textInput, 'flex-1', {
                        'bg-gray-100 cursor-not-allowed': isDownloading,
                        'bg-white': !isDownloading,
                      })}
                      type="text"
                      placeholder="Enter destination directory"
                      value={hfLocalDirDownload || ''}
                      onChange={(e) => setHfLocalDirDownload(e.target.value)}
                      disabled={isDownloading}
                    />
                  </div>
                  <div className="text-xs text-gray-500 flex items-center gap-1">
                    <MdFolderOpen className="inline-block w-4 h-4 text-blue-700 mr-1" />
                    Repo will land at{' '}
                    <span className="font-mono text-blue-700">
                      {(hfLocalDirDownload || '').replace(/\/$/, '')}/
                      {resolveFullRepoId(hfRepoIdDownload) || '<repo-id>'}
                    </span>
                  </div>
                </div>

                {/* Download Button */}
                <div className="w-full flex flex-row items-center justify-start gap-3 mt-2">
                  <button
                    className={clsx(
                      'px-6',
                      'py-2',
                      'text-sm',
                      'font-medium',
                      'rounded-lg',
                      'transition-colors',
                      {
                        'bg-blue-500 text-white hover:bg-blue-600': downloadButtonEnabled,
                        'bg-gray-300 text-gray-500 cursor-not-allowed': !downloadButtonEnabled,
                      }
                    )}
                    onClick={operations.downloadDataset}
                    disabled={!downloadButtonEnabled}
                  >
                    <div className="flex items-center justify-center gap-2">
                      <MdOutlineFileDownload className="w-6 h-6" />
                      Download
                    </div>
                  </button>

                  {/* Cancel Button */}
                  <button
                    className={clsx(STYLES.cancelButton, {
                      'bg-red-500 text-white hover:bg-red-600': isDownloading,
                      'bg-gray-300 text-gray-500 cursor-not-allowed': !isDownloading,
                    })}
                    onClick={operations.cancelOperation}
                    disabled={!isDownloading}
                  >
                    Cancel
                  </button>

                  {/* Status */}
                  <div className="flex flex-row items-center justify-start gap-2">
                    <span className="text-sm text-gray-500">
                      {isDownloading && '⏳ Downloading...'}
                      {!isDownloading && hfStatus}
                    </span>
                    {isDownloading && (
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* File Browser Modals for Dataset*/}
      <FileBrowserModal
        isOpen={showHfLocalDirBrowserModal}
        onClose={() => setShowHfLocalDirBrowserModal(false)}
        onFileSelect={handleHfLocalDirSelect}
        title="Select Local Directory for Upload"
        selectButtonText="Select"
        allowDirectorySelect={true}
        // Datasets are now stored as rosbag2 task folders, so we drop the
        // LeRobot marker (`meta` / `videos` / `data`) folder filter and just
        // start the picker at the rosbag2 root.
        initialPath={DEFAULT_PATHS.ROSBAG2_PATH}
        defaultPath={DEFAULT_PATHS.ROSBAG2_PATH}
        homePath=""
      />

      {/* File Browser Modal: download destination */}
      <FileBrowserModal
        isOpen={showHfDownloadDirBrowserModal}
        onClose={() => setShowHfDownloadDirBrowserModal(false)}
        onFileSelect={(item) => {
          setHfLocalDirDownload(item.full_path || item.path || '');
          setShowHfDownloadDirBrowserModal(false);
        }}
        title="Select destination directory for download"
        selectButtonText="Select"
        allowDirectorySelect={true}
        allowFileSelect={false}
        initialPath={
          downloadType === 'model'
            ? DEFAULT_PATHS.HF_MODEL_DOWNLOAD_PATH
            : DEFAULT_PATHS.HF_DATASET_DOWNLOAD_PATH
        }
        defaultPath={
          downloadType === 'model'
            ? DEFAULT_PATHS.HF_MODEL_DOWNLOAD_PATH
            : DEFAULT_PATHS.HF_DATASET_DOWNLOAD_PATH
        }
        homePath=""
      />

      {/* Token Input Popup */}
      <TokenInputPopup
        isOpen={showTokenPopup}
        onClose={() => setShowTokenPopup(false)}
        onSubmit={handleTokenSubmit}
        isLoading={isLoading}
        endpoint={hfActiveEndpoint}
        defaultLabel={
          (HF_ENDPOINT_PRESETS.find((p) => p.url === hfActiveEndpoint) || {}).label || ''
        }
      />
    </div>
  );
};

export default HuggingfaceSection;
