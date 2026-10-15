"""Preserve BT cursors while pausing leaves; never reset/reload on resume."""
import time


def leaves(root):
    if hasattr(root, 'children'):
        for child in root.children:
            yield from leaves(child)
    else:
        yield root


def pause_tree(root):
    actions = list(leaves(root))
    unsupported = [a.name for a in actions if not callable(getattr(a, 'pause', None))]
    if unsupported:
        raise RuntimeError('Pause unsupported by: ' + ', '.join(unsupported))
    # Pause every started asynchronous leaf, not only the displayed active UID.
    # Completed/unstarted actions implement pause as a no-op.
    errors = []
    for action in actions:
        try:
            action.pause()
        except Exception as exc:
            errors.append(f'{action.name}: {exc}')
    if errors:
        raise RuntimeError('; '.join(errors))


def resume_tree(root):
    for action in leaves(root):
        action.resume()


class TimerPause:
    """Freeze wall-clock deadlines while the executor is paused."""
    pause_timer_fields = ('_start_time',)

    def pause(self):
        if getattr(self, '_paused_at', None) is None:
            self._paused_at = time.monotonic()

    def resume(self):
        paused = getattr(self, '_paused_at', None)
        if paused is None:
            return
        duration = time.monotonic() - paused
        for field in self.pause_timer_fields:
            value = getattr(self, field, None)
            if value is not None:
                setattr(self, field, value + duration)
        self._paused_at = None

    def reset(self):
        self._paused_at = None
        super().reset()
