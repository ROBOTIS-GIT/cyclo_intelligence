import { serializeFromGraph } from './btXmlSerializer';
import { parseBTXml } from './btTreeParser';

function sampleGraph() {
  const nodes = [
    {
      id: 'bt_0',
      type: 'btControl',
      position: { x: 10, y: 20 },
      data: {
        label: 'Sequence_1',
        nodeType: 'Sequence',
        params: {},
        collapsed: false,
      },
    },
    {
      id: 'bt_1',
      type: 'btAction',
      position: { x: 30, y: 40 },
      data: {
        label: 'Wait_1',
        nodeType: 'Wait',
        params: { duration: '3.0' },
      },
    },
  ];
  const edges = [
    { id: 'e_bt_0_bt_1', source: 'bt_0', target: 'bt_1' },
  ];
  const nodeDataMap = new Map([
    ['bt_0', { tag: 'Sequence', name: 'Sequence_1', params: {}, collapsed: false }],
    ['bt_1', { tag: 'Wait', name: 'Wait_1', params: { duration: '3.0' } }],
  ]);
  return { nodes, edges, nodeDataMap };
}

describe('btXmlSerializer', () => {
  it('serializes graph nodes without layout attributes', () => {
    const { nodes, edges, nodeDataMap } = sampleGraph();
    const xml = serializeFromGraph(nodes, edges, nodeDataMap);

    expect(xml).toContain('<BehaviorTree ID="MainTree">');
    expect(xml).toContain('<Sequence name="Sequence_1"');
    expect(xml).toContain('<Wait name="Wait_1"');
    expect(xml).toContain('duration="3.0"');
    expect(xml).not.toContain('bt_x=');
    expect(xml).not.toContain('bt_y=');
  });

  it('round-trips serialized graphs through the parser', () => {
    const { nodes, edges, nodeDataMap } = sampleGraph();
    const xml = serializeFromGraph(nodes, edges, nodeDataMap);
    const parsed = parseBTXml(xml);

    expect(parsed.nodes).toHaveLength(2);
    expect(parsed.edges).toHaveLength(1);
    expect([...parsed.nodeDataMap.values()].map((entry) => entry.tag)).toEqual([
      'Sequence',
      'Wait',
    ]);
  });

  it('serializes SendCommand with command-specific parameters only', () => {
    const nodes = [
      {
        id: 'bt_0',
        type: 'btControl',
        position: { x: 0, y: 0 },
        data: { label: 'Sequence_1', nodeType: 'Sequence', params: {} },
      },
      {
        id: 'bt_1',
        type: 'btAction',
        position: { x: 100, y: 0 },
        data: {
          label: 'ResumeInference',
          nodeType: 'SendCommand',
          params: {
            command: 'RESUME',
            model: 'groot:n17',
            policy_path: '/workspace/model/groot/test',
            task_instruction: '',
            inference_mode: 'robot',
            action_request_mode: 'sync',
            inference_hz: '10',
            control_hz: '100',
            chunk_align_window_s: '0.3',
          },
        },
      },
      {
        id: 'bt_2',
        type: 'btAction',
        position: { x: 200, y: 0 },
        data: {
          label: 'StopInference',
          nodeType: 'SendCommand',
          params: {
            command: 'STOP',
            model: 'groot:n17',
            policy_path: '',
            task_instruction: '',
            action_request_mode: 'sync',
            inference_hz: '10',
            control_hz: '100',
            chunk_align_window_s: '0.3',
          },
        },
      },
    ];
    const edges = [
      { id: 'e_bt_0_bt_1', source: 'bt_0', target: 'bt_1' },
      { id: 'e_bt_0_bt_2', source: 'bt_0', target: 'bt_2' },
    ];
    const nodeDataMap = new Map([
      ['bt_0', { tag: 'Sequence', name: 'Sequence_1', params: {} }],
      ['bt_1', { tag: 'SendCommand', name: 'ResumeInference', params: nodes[1].data.params }],
      ['bt_2', { tag: 'SendCommand', name: 'StopInference', params: nodes[2].data.params }],
    ]);

    const xml = serializeFromGraph(nodes, edges, nodeDataMap);

    expect(xml).toContain('command="RESUME"');
    expect(xml).toContain('command="STOP"');
    expect(xml).not.toContain('bt_x=');
    expect(xml).not.toContain('bt_y=');
    expect(xml).not.toContain('model="groot:n17"');
    expect(xml).not.toContain('policy_path=');
    expect(xml).not.toContain('inference_mode=');
    expect(xml).not.toContain('action_request_mode=');
    expect(xml).not.toContain('inference_hz=');
    expect(xml).not.toContain('control_hz=');
    expect(xml).not.toContain('chunk_align_window_s=');
    expect(xml).not.toContain('acceleration_mode=');
    expect(xml).not.toContain('acceleration_engine_path=');
  });
});
