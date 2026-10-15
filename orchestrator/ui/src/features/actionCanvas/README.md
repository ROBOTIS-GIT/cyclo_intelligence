# Action Canvas

Action Canvas edits and executes a complete behavior tree. Its workspace and
`components/BTEditorSurface` compose reusable BT graph tools with Action-specific
persistence and runtime adapters.

| Module | Responsibility |
| --- | --- |
| `hooks/useActionBtPersistence` | Redux document hydration/persistence |
| `hooks/useActionBtFiles` | Tree library, loading, save-as, and conflicts |
| `hooks/useActionBtHistory` | Action document history adapter |
| `hooks/useActionBtExecution` | Start/stop sequencing and command checks |
| `hooks/useBtEngineLifecycle` | BT engine lifecycle/status |
| `hooks/useBtRuntimeSubscription` | Status and active-node subscriptions/cleanup |
| `components/ActionBt*` | Header, graph, dialogs, and runtime controls |
| `../btEditor` | Shared graph state/commands, deletion, rendering, and parameters |

Mission Canvas uses the same graph primitives through its own document adapter.
Sharing graph editing must never cause Mission editing to execute an Action tree,
or make either editor mutate the other editor's selection/history.

The older `components/navigation/ActionCanvasWorkspace` import is a compatibility
entry. New Action-specific behavior belongs in this feature. Existing Redux keys,
BT XML formats, and ROS start/stop calls are preserved.
