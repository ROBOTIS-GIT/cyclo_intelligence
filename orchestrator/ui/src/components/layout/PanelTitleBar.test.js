import { fireEvent, render, screen } from '@testing-library/react';
import PanelTitleBar from './PanelTitleBar';

describe('PanelTitleBar', () => {
  test('shows a collapse control when expanded', () => {
    const onToggleExpand = jest.fn();

    render(
      <PanelTitleBar
        title="Joint Data"
        expanded
        onToggleExpand={onToggleExpand}
        onClose={jest.fn()}
        onMouseDown={jest.fn()}
        isDragging={false}
      />
    );

    const collapseButton = screen.getByRole('button', { name: 'Collapse Joint Data' });
    expect(collapseButton).toBeInTheDocument();
    expect(collapseButton).toHaveAttribute('title', 'Collapse Joint Data');

    fireEvent.click(collapseButton);
    expect(onToggleExpand).toHaveBeenCalledTimes(1);
  });

  test('shows an expand control when collapsed', () => {
    render(
      <PanelTitleBar
        title="3D Viewer"
        expanded={false}
        onToggleExpand={jest.fn()}
        onClose={jest.fn()}
        onMouseDown={jest.fn()}
        isDragging={false}
      />
    );

    expect(screen.getByRole('button', { name: 'Expand 3D Viewer' })).toBeInTheDocument();
  });

  test('hides the expand control when the panel is not expandable', () => {
    render(
      <PanelTitleBar
        title="Cameras"
        expanded={false}
        onToggleExpand={jest.fn()}
        onClose={jest.fn()}
        onMouseDown={jest.fn()}
        isDragging={false}
        canExpand={false}
      />
    );

    expect(screen.queryByRole('button', { name: 'Expand Cameras' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Close Cameras' })).toBeInTheDocument();
  });
});
