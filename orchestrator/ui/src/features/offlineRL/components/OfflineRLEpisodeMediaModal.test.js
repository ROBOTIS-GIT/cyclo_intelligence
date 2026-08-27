import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import OfflineRLEpisodeMediaModal, {
  normalizeEpisodeJointData,
} from './OfflineRLEpisodeMediaModal';

const episode = {
  index: 7,
  outcome: 'success',
  frames: 45,
  tasks: ['Pick up the jelly bag'],
};

const media = [
  {
    key: 'left-wrist',
    label: 'Left wrist',
    url: '/media/left.mp4',
    fromS: 2,
    toS: 5,
    fps: 15,
  },
  {
    key: 'head',
    label: 'Head',
    url: '/media/head.mp4',
    fromS: 10,
    toS: 13,
    fps: 15,
  },
  {
    key: 'right-wrist',
    label: 'Right wrist',
    url: '/media/right.mp4',
    fromS: 20,
    toS: 23,
    fps: 15,
  },
];

const defaultProps = {
  open: true,
  sourceLabel: 'LeRobot v3.0',
  episode,
  media,
  onBack: jest.fn(),
  onDelete: jest.fn(),
};

const renderModal = (props = {}) => render(
  <OfflineRLEpisodeMediaModal {...defaultProps} {...props} />
);

const makeVideoStateWritable = (video, { currentTime = 0, paused = true } = {}) => {
  Object.defineProperty(video, 'currentTime', {
    configurable: true,
    writable: true,
    value: currentTime,
  });
  Object.defineProperty(video, 'paused', {
    configurable: true,
    writable: true,
    value: paused,
  });
};

describe('OfflineRLEpisodeMediaModal', () => {
  let playSpy;
  let pauseSpy;

  beforeEach(() => {
    defaultProps.onBack = jest.fn();
    defaultProps.onDelete = jest.fn();
    playSpy = jest.spyOn(HTMLMediaElement.prototype, 'play')
      .mockImplementation(function play() {
        this.paused = false;
        return Promise.resolve();
      });
    pauseSpy = jest.spyOn(HTMLMediaElement.prototype, 'pause')
      .mockImplementation(function pause() {
        this.paused = true;
      });
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('does not render while closed', () => {
    renderModal({ open: false });

    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
  });

  test('shows episode metadata and three camera videos', () => {
    const { container } = renderModal();

    const dialog = screen.getByRole('dialog', { name: 'episode_007' });
    const backdrop = screen.getByTestId('offline-rl-episode-media-backdrop');
    expect(dialog).toBeInTheDocument();
    expect(backdrop.parentElement).toBe(document.body);
    expect(container).not.toContainElement(backdrop);
    expect(dialog).toHaveClass('h-[86vh]', 'w-[98vw]', 'max-w-none');
    expect(screen.getAllByText('LeRobot v3.0').length).toBeGreaterThan(0);
    expect(screen.getByText('Success')).toBeInTheDocument();
    expect(screen.getByText('15')).toBeInTheDocument();
    expect(screen.getAllByText('3.00 s').length).toBeGreaterThan(0);
    expect(screen.getByText('45')).toBeInTheDocument();
    expect(screen.getByText('Pick up the jelly bag')).toBeInTheDocument();

    const videos = screen.getAllByLabelText(/episode video$/);
    expect(videos).toHaveLength(3);
    expect(videos[0]).toHaveAttribute('src', '/media/left.mp4');
    expect(videos[1]).toHaveAttribute('src', '/media/head.mp4');
    expect(videos[2]).toHaveAttribute('src', '/media/right.mp4');
    expect(videos[0]).toHaveAttribute('preload', 'metadata');
  });

  test('shows Replay Viewer joint state/action values at the video time', () => {
    renderModal({
      jointData: {
        joint_timestamps: [0, 1],
        joint_names: ['arm_l_joint1'],
        joint_positions: [0.1, 0.2],
        action_timestamps: [0, 1],
        action_names: ['arm_l_joint1'],
        action_values: [0.15, 0.25],
        duration: 1,
      },
    });
    const videos = screen.getAllByLabelText(/episode video$/);
    videos.forEach((video) => makeVideoStateWritable(video));
    videos[0].currentTime = 3;
    fireEvent.timeUpdate(videos[0]);

    expect(screen.getByRole('region', { name: 'Episode joint data' }))
      .toHaveClass('h-[460px]', 'overflow-hidden');
    expect(screen.getByText('Joint Data')).toBeInTheDocument();
    expect(screen.getByText('arm_l_joint1')).toBeInTheDocument();
    expect(screen.getByText('0.2000')).toBeInTheDocument();
    expect(screen.getByText('0.2500')).toBeInTheDocument();
  });

  test('locks page scrolling while open and restores it when closed', () => {
    document.body.style.overflow = 'auto';
    const { rerender } = renderModal();

    expect(document.body.style.overflow).toBe('hidden');

    rerender(
      <OfflineRLEpisodeMediaModal
        {...defaultProps}
        open={false}
      />
    );
    expect(document.body.style.overflow).toBe('auto');
    document.body.style.overflow = '';
  });

  test('starts each video at its segment boundary and synchronizes play, seek, and pause', async () => {
    renderModal();
    const videos = screen.getAllByLabelText(/episode video$/);
    videos.forEach((video) => makeVideoStateWritable(video));

    videos.forEach((video) => fireEvent.loadedMetadata(video));
    expect(videos.map((video) => video.currentTime)).toEqual([2, 10, 20]);

    videos[0].paused = false;
    videos[0].currentTime = 3;
    fireEvent.play(videos[0]);
    expect(videos[1].currentTime).toBe(11);
    expect(videos[2].currentTime).toBe(21);
    expect(playSpy).toHaveBeenCalledTimes(2);

    videos[0].currentTime = 4.4;
    fireEvent.seeked(videos[0]);
    expect(videos[1].currentTime).toBeCloseTo(12.4);
    expect(videos[2].currentTime).toBeCloseTo(22.4);

    videos[0].paused = true;
    fireEvent.pause(videos[0]);
    expect(videos[1].paused).toBe(true);
    expect(videos[2].paused).toBe(true);
    expect(pauseSpy).toHaveBeenCalledTimes(2);

    videos[0].currentTime = 0;
    fireEvent.seeked(videos[0]);
    expect(videos[0].currentTime).toBe(2);
    expect(videos[1].currentTime).toBe(10);
    expect(videos[2].currentTime).toBe(20);

    await act(async () => Promise.resolve());
  });

  test('stops all cameras at the configured segment end', () => {
    renderModal();
    const videos = screen.getAllByLabelText(/episode video$/);
    videos.forEach((video, index) => makeVideoStateWritable(video, {
      currentTime: [5, 13, 23][index],
      paused: false,
    }));

    fireEvent.timeUpdate(videos[0]);

    expect(videos[0].currentTime).toBe(5);
    expect(videos.every((video) => video.paused)).toBe(true);
  });

  test('confirms deletion and closes only after a successful delete', async () => {
    const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(true);
    const onDelete = jest.fn().mockResolvedValue(undefined);
    const onBack = jest.fn();
    renderModal({ onDelete, onBack });

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));

    expect(confirmSpy).toHaveBeenCalledWith(
      'Delete episode_007? This removes the selected episode from its dataset.'
    );
    expect(screen.getByRole('button', { name: 'Deleting…' })).toBeDisabled();
    await waitFor(() => expect(onDelete).toHaveBeenCalledWith(episode));
    await waitFor(() => expect(onBack).toHaveBeenCalledTimes(1));
  });

  test('keeps the dialog open when deletion is cancelled, declined, or fails', async () => {
    const confirmSpy = jest.spyOn(window, 'confirm').mockReturnValue(false);
    const onDelete = jest.fn();
    const onBack = jest.fn();
    const { rerender } = renderModal({ onDelete, onBack });

    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(onDelete).not.toHaveBeenCalled();
    expect(onBack).not.toHaveBeenCalled();

    confirmSpy.mockReturnValue(true);
    onDelete.mockResolvedValueOnce(false);
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The episode could not be deleted.'
    );
    expect(onBack).not.toHaveBeenCalled();

    onDelete.mockRejectedValueOnce(new Error('Dataset is locked'));
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Dataset is locked');
    expect(onBack).not.toHaveBeenCalled();

    rerender(
      <OfflineRLEpisodeMediaModal
        {...defaultProps}
        onDelete={onDelete}
        onBack={onBack}
        deletePending
      />
    );
    expect(screen.getByRole('button', { name: 'Deleting…' })).toBeDisabled();
  });

  test('supports loading, API error, and empty-media states', () => {
    const { rerender } = renderModal({ loading: true });
    expect(screen.getByRole('status')).toHaveTextContent('Loading episode media…');

    rerender(
      <OfflineRLEpisodeMediaModal
        {...defaultProps}
        loading={false}
        media={[]}
        error="Episode media index is unavailable"
      />
    );
    expect(screen.getByRole('alert')).toHaveTextContent(
      'Episode media index is unavailable'
    );
    expect(screen.getByText(
      'No playable camera video is available for this episode.'
    )).toBeInTheDocument();
  });

  test('closes on Back, Escape, and backdrop while restoring focus', () => {
    const opener = document.createElement('button');
    document.body.appendChild(opener);
    opener.focus();
    const onBack = jest.fn();
    const { rerender } = renderModal({ onBack });
    const backButton = screen.getByRole('button', { name: 'Back to episode list' });

    expect(backButton).toHaveFocus();
    fireEvent.keyDown(window, { key: 'Escape' });
    fireEvent.mouseDown(screen.getByRole('dialog'));
    fireEvent.mouseDown(screen.getByTestId('offline-rl-episode-media-backdrop'));
    fireEvent.click(backButton);
    expect(onBack).toHaveBeenCalledTimes(3);

    rerender(
      <OfflineRLEpisodeMediaModal
        {...defaultProps}
        open={false}
        onBack={onBack}
      />
    );
    expect(opener).toHaveFocus();
    opener.remove();
  });
});

test('normalizes snake-case and camel-case episode joint contracts', () => {
  expect(normalizeEpisodeJointData({
    joint_timestamps: [0],
    joint_names: ['joint_1'],
    joint_positions: [0.5],
    actionTimestamps: [0],
    actionNames: ['joint_1'],
    actionValues: [0.6],
    duration: 1,
  })).toEqual({
    jointTimestamps: [0],
    jointNames: ['joint_1'],
    jointPositions: [0.5],
    actionTimestamps: [0],
    actionNames: ['joint_1'],
    actionValues: [0.6],
    duration: 1,
  });
});
