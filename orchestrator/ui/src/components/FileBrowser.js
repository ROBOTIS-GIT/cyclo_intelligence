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
// Author: Kiwoong Park, Seongwoo Kim

import React, { useCallback, useEffect, useState, useRef } from 'react';
import clsx from 'clsx';
import toast from 'react-hot-toast';
import {
  MdRefresh,
  MdFolder,
  MdDescription,
  MdHome,
  MdArrowUpward,
  MdKeyboardArrowRight,
  MdCheck,
  MdStar,
  MdBookmark,
} from 'react-icons/md';
import { useRosServiceCaller } from '../hooks/useRosServiceCaller';

/**
 * Format file size in bytes to human readable format
 */
const formatFileSize = (bytes) => {
  if (bytes < 0) return '-'; // Directory
  if (bytes === 0) return '0 B';

  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
};

const filterItems = (items, targetFileName, fileFilter) => {
  if (targetFileName) {
    return items.filter((item) => item.is_directory);
  } else if (fileFilter) {
    return items.filter((item) => item.is_directory || fileFilter(item));
  }
  return items;
};
const hasTargetFile = (item, targetFileName, targetFolderName, directoriesWithTarget) => {
  return (
    (targetFileName || targetFolderName) &&
    item.is_directory &&
    directoriesWithTarget.has(item.full_path)
  );
};

const FileBrowserHeader = ({
  title,
  onGoHome,
  onGoDefault,
  onGoParent,
  onRefresh,
  loading,
  parentPath,
  homePath,
  defaultPath,
  variant = 'legacy',
}) => {
  const autonomyStudioVariant = variant === 'autonomy-studio';
  const classHeader = clsx(
    'flex',
    'items-center',
    'justify-between',
    'p-4',
    'border-b',
    autonomyStudioVariant ? 'border-[var(--mc-border)]' : 'border-gray-200'
  );

  const classTitle = clsx(
    'font-semibold',
    autonomyStudioVariant ? 'text-sm text-[var(--mc-text)]' : 'text-lg text-gray-900',
  );

  const classButtonContainer = clsx('flex', 'items-center', 'space-x-2');

  const classButton = clsx(
    'p-2',
    autonomyStudioVariant
      ? 'text-[var(--mc-text-muted)] hover:text-[var(--mc-accent-hover)] hover:bg-[var(--mc-surface-hover)]'
      : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50',
    'rounded-lg',
    'transition-colors'
  );

  const classButtonWithDisabled = clsx(
    classButton,
    'disabled:opacity-50',
    'disabled:cursor-not-allowed'
  );

  return (
    <div className={classHeader}>
      <h3 className={classTitle}>{title}</h3>
      <div className={classButtonContainer}>
        <button
          onClick={onGoHome}
          disabled={loading}
          className={classButton}
          title={homePath ? `Home: ${homePath}` : 'Home'}
        >
          <MdHome size={20} />
        </button>
        {defaultPath && (
          <button
            onClick={onGoDefault}
            disabled={loading}
            className={classButton}
            title={`Default: ${defaultPath}`}
          >
            <MdBookmark size={20} />
          </button>
        )}
        <button
          onClick={onGoParent}
          disabled={loading || !parentPath}
          className={classButtonWithDisabled}
          title="Parent Directory"
        >
          <MdArrowUpward size={20} />
        </button>
        <button
          onClick={onRefresh}
          disabled={loading}
          className={clsx(classButton, 'disabled:opacity-50')}
          title="Refresh"
        >
          <MdRefresh size={20} className={clsx(loading && 'animate-spin')} />
        </button>
      </div>
    </div>
  );
};

const PathInfo = ({
  currentPath,
  homePath,
  defaultPath,
  targetFileName,
  variant = 'legacy',
}) => {
  const autonomyStudioVariant = variant === 'autonomy-studio';
  const classContainer = clsx(
    'px-4',
    'py-2',
    'border-b',
    autonomyStudioVariant
      ? 'bg-[var(--mc-surface-2)] border-[var(--mc-border)]'
      : 'bg-gray-50 border-gray-200',
  );

  const classPathRow = clsx(
    'flex',
    'items-center',
    'text-sm',
    autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-600',
  );

  const classLabel = clsx('font-medium');

  const classPathValue = clsx('ml-2', 'font-mono', 'break-all');

  const classHomeRow = clsx(
    'flex',
    'items-center',
    'mt-1',
    'text-xs',
    autonomyStudioVariant ? 'text-[var(--mc-accent)]' : 'text-purple-600',
  );

  const classHomeBadge = clsx(
    'ml-2',
    'font-mono',
    'px-2',
    'py-1',
    'rounded',
    autonomyStudioVariant ? 'bg-[var(--mc-accent-soft)]' : 'bg-purple-100',
  );

  const classDefaultRow = clsx(
    'flex',
    'items-center',
    'mt-1',
    'text-xs',
    autonomyStudioVariant ? 'text-[var(--mc-warning)]' : 'text-orange-600',
  );

  const classDefaultBadge = clsx(
    'ml-2',
    'font-mono',
    'px-2',
    'py-1',
    'rounded',
    autonomyStudioVariant ? 'bg-[var(--mc-surface-hover)]' : 'bg-orange-100',
  );

  return (
    <div className={classContainer}>
      <div className={classPathRow}>
        <span className={classLabel}>Path:</span>
        <span className={classPathValue}>{currentPath || '/'}</span>
      </div>
      {homePath && (
        <div className={classHomeRow}>
          <span className={classLabel}>Home:</span>
          <span className={classHomeBadge}>{homePath}</span>
        </div>
      )}
      {defaultPath && (
        <div className={classDefaultRow}>
          <span className={classLabel}>
            <MdBookmark size={20} />
          </span>
          <span className={classDefaultBadge}>{defaultPath}</span>
        </div>
      )}
      {/* {targetFileName && (
        <div className={classTargetRow}>
          <span className={classLabel}>Looking for:</span>
          <span className={classTargetBadge}>{targetFileName}</span>
          <span className="ml-2">in directories</span>
        </div>
      )} */}
    </div>
  );
};

const FileItem = ({
  item,
  isSelected,
  hasTarget,
  targetFileName,
  targetFileLabel,
  onClick,
  onSelect,
  allowDirectorySelect = false,
  allowFileSelect = true,
  variant = 'legacy',
}) => {
  const autonomyStudioVariant = variant === 'autonomy-studio';
  const classItemContainer = clsx(
    'flex',
    'items-center',
    'px-4',
    'py-3',
    'cursor-pointer',
    'transition-colors',
    autonomyStudioVariant
      ? hasTarget
        ? 'bg-[var(--mc-surface-2)] hover:bg-[var(--mc-surface-hover)]'
        : 'hover:bg-[var(--mc-surface-hover)]'
      : hasTarget ? 'hover:bg-green-50 bg-green-25' : 'hover:bg-blue-50',
    isSelected &&
      (autonomyStudioVariant
        ? 'bg-[var(--mc-accent-soft)] border-l-4 border-[var(--mc-accent)]'
        : hasTarget
          ? 'bg-green-100 border-l-4 border-green-500'
          : 'bg-blue-100 border-l-4 border-blue-500')
  );

  const classIconContainer = clsx('flex-shrink-0', 'mr-3', 'relative');

  const classFolderIcon = clsx(
    'w-5',
    'h-5',
    autonomyStudioVariant
      ? hasTarget ? 'text-[var(--mc-success)]' : 'text-[var(--mc-accent)]'
      : hasTarget ? 'text-green-500' : 'text-blue-500',
  );

  const classStarIcon = clsx(
    'w-3',
    'h-3',
    autonomyStudioVariant ? 'text-[var(--mc-warning)]' : 'text-yellow-500',
    'absolute',
    '-top-1',
    '-right-1',
  );

  const classFileIcon = clsx(
    'w-5',
    'h-5',
    autonomyStudioVariant ? 'text-[var(--mc-text-subtle)]' : 'text-gray-400',
  );

  const classContentContainer = clsx('flex-1', 'min-w-0');

  const classHeaderRow = clsx('flex', 'items-center', 'justify-between');

  const classNameContainer = clsx('flex', 'items-center');

  const classItemName = clsx(
    'text-sm',
    'font-medium',
    'truncate',
    autonomyStudioVariant
      ? 'text-[var(--mc-text)]'
      : hasTarget ? 'text-green-900' : 'text-gray-900'
  );

  const classTargetBadge = clsx(
    'ml-2',
    'text-xs',
    autonomyStudioVariant
      ? 'bg-[var(--mc-surface-hover)] text-[var(--mc-success)]'
      : 'bg-green-100 text-green-700',
    'px-2',
    'py-1',
    'rounded-full'
  );

  const classCheckIcon = clsx(
    'w-4',
    'h-4',
    'ml-2',
    autonomyStudioVariant
      ? hasTarget ? 'text-[var(--mc-success)]' : 'text-[var(--mc-accent)]'
      : hasTarget ? 'text-green-600' : 'text-blue-600',
  );

  const classMetaRow = clsx(
    'flex',
    'items-center',
    'mt-1',
    'text-xs',
    autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-500',
  );

  const classArrowIcon = clsx(
    'w-4',
    'h-4',
    'ml-2',
    autonomyStudioVariant ? 'text-[var(--mc-text-subtle)]' : 'text-gray-400',
  );

  const canSelectDirectory = item.is_directory && allowDirectorySelect;
  const canSelectFile = !item.is_directory && allowFileSelect;
  const showSelectButton = canSelectDirectory || canSelectFile;

  const classSelectButton = clsx(
    'flex-shrink-0',
    'ml-2',
    'px-3',
    'py-1',
    'text-xs',
    'font-medium',
    'rounded-md',
    'border',
    'transition-colors',
    autonomyStudioVariant
      ? isSelected
        ? 'bg-[var(--mc-accent)] text-[var(--mc-accent-fg)] border-[var(--mc-accent)]'
        : 'bg-[var(--mc-surface)] text-[var(--mc-accent)] border-[var(--mc-border-strong)] hover:bg-[var(--mc-surface-hover)]'
      : isSelected
        ? 'bg-blue-600 text-white border-blue-600'
        : 'bg-white text-blue-600 border-blue-300 hover:bg-blue-50'
  );

  const handleItemClick = (e) => {
    // Only trigger navigation when the select button is not clicked
    if (!e.target.closest('.select-button')) {
      onClick(item);
    }
  };

  const handleSelectClick = (e) => {
    e.stopPropagation();
    if (onSelect) {
      onSelect(item);
    }
  };

  return (
    <div onClick={handleItemClick} className={classItemContainer}>
      <div className={classIconContainer}>
        {item.is_directory ? (
          <>
            <MdFolder className={classFolderIcon} />
            {hasTarget && <MdStar className={classStarIcon} />}
          </>
        ) : (
          <MdDescription className={classFileIcon} />
        )}
      </div>

      <div className={classContentContainer}>
        <div className={classHeaderRow}>
          <div className={classNameContainer}>
            <p className={classItemName}>{item.name}</p>
            {hasTarget && (
              <span className={classTargetBadge}>
                {targetFileLabel || `Contains ${targetFileName}`}
              </span>
            )}
          </div>

          <div className="flex items-center">
            {showSelectButton && (
              <button
                onClick={handleSelectClick}
                className={`${classSelectButton} select-button`}
                title={`Select this ${item.is_directory ? 'folder' : 'file'}`}
              >
                {isSelected ? 'Selected' : 'Select'}
              </button>
            )}
            {isSelected && <MdCheck className={classCheckIcon} />}
          </div>
        </div>
        <div className={classMetaRow}>
          <span>{formatFileSize(item.size)}</span>
          <span className="mx-2">•</span>
          <span>{item.modified_time}</span>
          {canSelectDirectory && (
            <>
              <span className="mx-2">•</span>
              <span className={clsx(
                'font-medium',
                autonomyStudioVariant ? 'text-[var(--mc-accent)]' : 'text-blue-600',
              )}>
                Selectable folder
              </span>
            </>
          )}
          {item.is_directory && !allowDirectorySelect && (
            <>
              <span className="mx-2">•</span>
              <span className={clsx(
                'font-medium',
                autonomyStudioVariant ? 'text-[var(--mc-text-subtle)]' : 'text-gray-400',
              )}>
                Navigation only
              </span>
            </>
          )}
          {!item.is_directory && !allowFileSelect && (
            <>
              <span className="mx-2">•</span>
              <span className={clsx(
                'font-medium',
                autonomyStudioVariant ? 'text-[var(--mc-text-subtle)]' : 'text-gray-400',
              )}>
                File selection disabled
              </span>
            </>
          )}
        </div>
      </div>

      {item.is_directory && !showSelectButton && (
        <MdKeyboardArrowRight className={classArrowIcon} />
      )}
    </div>
  );
};

const LoadingState = ({ variant = 'legacy' }) => {
  const autonomyStudioVariant = variant === 'autonomy-studio';
  const classContainer = clsx('flex', 'items-center', 'justify-center', 'py-8');

  const classSpinner = clsx(
    'animate-spin',
    'rounded-full',
    'h-6',
    'w-6',
    'border-b-2',
    autonomyStudioVariant ? 'border-[var(--mc-accent)]' : 'border-blue-600'
  );

  const classText = clsx(
    'ml-2',
    autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-600',
  );

  return (
    <div className={classContainer}>
      <div className={classSpinner}></div>
      <span className={classText}>Loading...</span>
    </div>
  );
};

const EmptyState = ({ variant = 'legacy' }) => {
  const autonomyStudioVariant = variant === 'autonomy-studio';
  const classContainer = clsx(
    'flex',
    'items-center',
    'justify-center',
    'py-8',
    autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-gray-500',
  );

  return (
    <div className={classContainer}>
      <span>No items found</span>
    </div>
  );
};

const ErrorDisplay = ({ error, variant = 'legacy' }) => {
  const autonomyStudioVariant = variant === 'autonomy-studio';
  const classContainer = clsx(
    'px-4',
    'py-3',
    'border-b',
    autonomyStudioVariant
      ? 'bg-[var(--mc-surface-2)] border-[var(--mc-danger-border)]'
      : 'bg-red-50 border-red-200',
  );

  const classText = clsx(
    'text-sm',
    autonomyStudioVariant ? 'text-[var(--mc-danger)]' : 'text-red-600',
  );

  return (
    <div className={classContainer}>
      <p className={classText}>{error}</p>
    </div>
  );
};
const SelectedItemInfo = ({
  selectedItem,
  targetFileName,
  directoriesWithTarget,
  targetFileLabel,
  variant = 'legacy',
}) => {
  if (!selectedItem) return null;

  const isTargetDirectory =
    targetFileName &&
    selectedItem.is_directory &&
    directoriesWithTarget.has(selectedItem.full_path);
  const autonomyStudioVariant = variant === 'autonomy-studio';

  const classContainer = clsx(
    'px-4',
    'py-3',
    'border-t',
    autonomyStudioVariant
      ? 'bg-[var(--mc-surface-2)] border-[var(--mc-border)]'
      : isTargetDirectory ? 'bg-green-50 border-green-200' : 'bg-blue-50 border-blue-200'
  );

  const classContent = clsx('text-sm');

  const classLabel = clsx(
    'font-medium',
    autonomyStudioVariant
      ? 'text-[var(--mc-text)]'
      : isTargetDirectory ? 'text-green-900' : 'text-blue-900',
  );

  const classPath = clsx(
    'font-mono',
    'break-all',
    'mt-1',
    autonomyStudioVariant
      ? 'text-[var(--mc-text-muted)]'
      : isTargetDirectory ? 'text-green-700' : 'text-blue-700'
  );

  const classTargetInfo = clsx(
    autonomyStudioVariant ? 'text-[var(--mc-success)]' : 'text-green-600',
    'text-xs',
    'mt-2',
    'flex',
    'items-center',
  );

  const classFileLabel = clsx(
    'font-medium',
    autonomyStudioVariant ? 'text-[var(--mc-text)]' : 'text-blue-900',
  );

  const classFilePath = clsx(
    autonomyStudioVariant ? 'text-[var(--mc-text-muted)]' : 'text-blue-700',
    'font-mono',
    'break-all',
    'mt-1',
  );

  const classStarIcon = clsx('w-3', 'h-3', 'mr-1');

  return (
    <div className={classContainer}>
      <div className={classContent}>
        {selectedItem.is_directory ? (
          <>
            <p className={classLabel}>Selected Directory:</p>
            <p className={classPath}>{selectedItem.full_path}</p>
            {isTargetDirectory && (
              <p className={classTargetInfo}>
                <MdStar className={classStarIcon} />
                {targetFileLabel || `This directory contains ${targetFileName}`}
              </p>
            )}
          </>
        ) : (
          <>
            <p className={classFileLabel}>Selected File:</p>
            <p className={classFilePath}>{selectedItem.full_path}</p>
          </>
        )}
      </div>
    </div>
  );
};

/**
 * FileBrowser - A comprehensive file browser component with navigation and selection capabilities
 */
export default function FileBrowser({
  onFileSelect,
  onPathChange,
  initialPath = '',
  fileFilter = null,
  className = '',
  title = 'File Browser',
  targetFileName = null,
  targetFolderName = null,
  onDirectorySelect = null,
  targetFileLabel = null,
  homePath = null,
  defaultPath = null,
  allowDirectorySelect = false,
  allowFileSelect = true,
  multiSelect = false,
  onSelectionChange = null,
  variant = 'legacy',
}) {
  const { browseFile } = useRosServiceCaller();
  const autonomyStudioVariant = variant === 'autonomy-studio';
  const isInitializedRef = useRef(false);

  const [currentPath, setCurrentPath] = useState(initialPath);
  const [parentPath, setParentPath] = useState('');
  const [items, setItems] = useState([]);
  const [selectedItem, setSelectedItem] = useState(null);
  const [selectedItems, setSelectedItems] = useState(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [directoriesWithTarget, setDirectoriesWithTarget] = useState(new Set());
  const checkDirectoriesForTargetFile = useCallback((itemList) => {
    // Server-side parallel checking now provides has_target_file field
    // No need for client-side checking anymore
    const newDirectoriesWithTarget = new Set();

    itemList.forEach((item) => {
      if (item.is_directory && item.has_target_file) {
        newDirectoriesWithTarget.add(item.full_path);
      }
    });

    setDirectoriesWithTarget(newDirectoriesWithTarget);
  }, []);

  const browsePath = useCallback(
    async (path, action = 'browse', targetName = '') => {
      setLoading(true);
      setError(null);
      setSelectedItem(null);

      try {
        // Only pass target files if we actually have a targetFileName
        const targetFiles = targetFileName ? targetFileName : null;
        const targetFolders = targetFolderName ? targetFolderName : null;
        const result = await browseFile(action, path, targetName, targetFiles, targetFolders);

        if (result.success) {
          setCurrentPath(result.current_path);
          setParentPath(result.parent_path);
          setItems(result.items || []);

          if (onPathChange) {
            onPathChange(result.current_path);
          }

          // Server already checked for target files, just process the results
          if (targetFileName && result.items) {
            checkDirectoriesForTargetFile(result.items);
          } else if (targetFolderName && result.items) {
            checkDirectoriesForTargetFile(result.items);
          } else if (!targetFileName && !targetFolderName) {
            setDirectoriesWithTarget(new Set());
          }
        } else {
          setError(result.message || 'Failed to browse directory');
          toast.error(result.message || 'Failed to browse directory');
        }
      } catch (err) {
        const errorMessage = err.message || 'Failed to browse directory';
        setError(errorMessage);
        toast.error(errorMessage);
      } finally {
        setLoading(false);
      }
    },
    [browseFile, onPathChange, targetFileName, targetFolderName, checkDirectoriesForTargetFile]
  );

  const goHome = useCallback(async () => {
    if (homePath) {
      await browsePath(homePath, 'browse');
    } else {
      await browsePath('', 'browse');
    }
  }, [browsePath, homePath]);

  const goDefault = useCallback(async () => {
    if (defaultPath) {
      await browsePath(defaultPath, 'browse');
    }
  }, [browsePath, defaultPath]);

  const goParent = useCallback(async () => {
    if (parentPath) {
      await browsePath(currentPath, 'go_parent');
    }
  }, [browsePath, currentPath, parentPath]);

  const refresh = useCallback(() => {
    browsePath(currentPath, 'browse');
    setSelectedItem(null);
  }, [browsePath, currentPath]);

  const handleItemClick = useCallback(
    async (item) => {
      if (item.is_directory) {
        // Special handling when a target is present
        if (targetFileName || targetFolderName) {
          if (item.has_target_file) {
            setSelectedItem(item);
            if (onDirectorySelect) {
              onDirectorySelect(item);
            } else if (onFileSelect) {
              onFileSelect(item);
            }
            return;
          }
        }

        // Normal folder navigation (not selected)
        await browsePath(item.full_path, 'browse');
        setSelectedItem(null);
      }
      // File click is handled in handleItemSelect
    },
    [browsePath, onFileSelect, onDirectorySelect, targetFileName, targetFolderName]
  );

  const handleItemSelect = useCallback(
    (item) => {
      if (multiSelect) {
        setSelectedItems((prev) => {
          const next = new Map(prev);
          if (next.has(item.full_path)) {
            next.delete(item.full_path);
          } else {
            next.set(item.full_path, item);
          }
          if (onSelectionChange) {
            onSelectionChange(Array.from(next.values()));
          }
          return next;
        });
      } else {
        setSelectedItem(item);

        if (item.is_directory) {
          if (onDirectorySelect) {
            onDirectorySelect(item);
          } else if (onFileSelect) {
            onFileSelect(item);
          }
        } else {
          if (onFileSelect) {
            onFileSelect(item);
          }
        }
      }
    },
    [multiSelect, onFileSelect, onDirectorySelect, onSelectionChange]
  );

  const filteredItems = filterItems(items, targetFileName, fileFilter);

  useEffect(() => {
    // Prevent multiple initializations
    if (isInitializedRef.current) {
      return;
    }
    isInitializedRef.current = true;

    const initializeBrowser = async () => {
      const targetPath = initialPath || homePath;
      setLoading(true);
      setError(null);
      setSelectedItem(null);

      try {
        const targetFiles = targetFileName ? targetFileName : null;
        const targetFolders = targetFolderName ? targetFolderName : null;
        const result = await browseFile('browse', targetPath || '', '', targetFiles, targetFolders);

        if (result.success) {
          setCurrentPath(result.current_path);
          setParentPath(result.parent_path);
          setItems(result.items || []);

          if (onPathChange) {
            onPathChange(result.current_path);
          }

          if (targetFileName && result.items) {
            checkDirectoriesForTargetFile(result.items);
          } else if (targetFolderName && result.items) {
            checkDirectoriesForTargetFile(result.items);
          } else if (!targetFileName && !targetFolderName) {
            setDirectoriesWithTarget(new Set());
          }
        } else {
          setError(result.message || 'Failed to browse directory');
          toast.error(result.message || 'Failed to browse directory');
        }
      } catch (err) {
        const errorMessage = err.message || 'Failed to browse directory';
        setError(errorMessage);
        toast.error(errorMessage);
      } finally {
        setLoading(false);
      }
    };

    initializeBrowser();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialPath, homePath, targetFileName, targetFolderName]);

  const classMainContainer = clsx(
    'h-full',
    'w-full',
    'flex',
    'flex-col',
    autonomyStudioVariant ? 'bg-[var(--mc-surface)] text-[var(--mc-text)]' : 'bg-white',
    'border',
    autonomyStudioVariant ? 'border-[var(--mc-border-strong)]' : 'border-gray-300',
    'rounded-lg',
    className
  );

  const classScrollContainer = clsx('overflow-y-auto', 'scrollbar-thin');

  const classItemList = clsx(
    'divide-y',
    autonomyStudioVariant ? 'divide-[var(--mc-border)]' : 'divide-gray-100',
  );

  return (
    <div className={classMainContainer}>
      <FileBrowserHeader
        title={title}
        onGoHome={goHome}
        onGoDefault={goDefault}
        onGoParent={goParent}
        onRefresh={refresh}
        loading={loading}
        parentPath={parentPath}
        homePath={homePath}
        defaultPath={defaultPath}
        variant={variant}
      />

      <PathInfo
        currentPath={currentPath}
        homePath={homePath}
        defaultPath={defaultPath}
        targetFileName={targetFileName}
        variant={variant}
      />

      {error && <ErrorDisplay error={error} variant={variant} />}

      {loading && <LoadingState variant={variant} />}

      {!loading && (
        <div className={classScrollContainer}>
          {filteredItems.length === 0 ? (
            <EmptyState variant={variant} />
          ) : (
            <div className={classItemList}>
              {filteredItems.map((item, index) => {
                const itemHasTarget = hasTargetFile(
                  item,
                  targetFileName,
                  targetFolderName,
                  directoriesWithTarget
                );
                const isSelected = multiSelect
                  ? selectedItems.has(item.full_path)
                  : selectedItem?.full_path === item.full_path;

                return (
                  <FileItem
                    key={item.full_path}
                    item={item}
                    isSelected={isSelected}
                    hasTarget={itemHasTarget}
                    targetFileName={targetFileName}
                    targetFileLabel={targetFileLabel}
                    onClick={handleItemClick}
                    onSelect={handleItemSelect}
                    allowDirectorySelect={allowDirectorySelect}
                    allowFileSelect={allowFileSelect}
                    variant={variant}
                  />
                );
              })}
            </div>
          )}
        </div>
      )}

      <SelectedItemInfo
        selectedItem={selectedItem}
        targetFileName={targetFileName}
        directoriesWithTarget={directoriesWithTarget}
        targetFileLabel={targetFileLabel}
        variant={variant}
      />
    </div>
  );
}
