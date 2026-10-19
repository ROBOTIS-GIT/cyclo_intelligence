import { parameterNodes, updateParameter } from './localBtParameters';
const xml = '<root main_tree_to_execute="MainTree"><BehaviorTree ID="MainTree"><Sequence name="Root"><!-- keep --><ArucoDock name="Dock" marker_id="11"/><ReturnToWaypoint name="Back" waypoint="a"/></Sequence></BehaviorTree></root>';
test('adds catalog defaults for missing fields and edits only the selected node', () => {
  const nodes = parameterNodes(xml, [{tag:'ArucoDock', ports:[{name:'marker_size',default:'.2'}]}]);
  expect(nodes[0].id).toBe('1');
  expect(nodes[0].data.params.marker_size).toBe('.2');
  const result = updateParameter(xml, '1', 'target_distance', '.55');
  const doc = new DOMParser().parseFromString(result, 'text/xml');
  expect(doc.querySelector('ArucoDock').getAttribute('target_distance')).toBe('.55');
  expect(doc.querySelector('ReturnToWaypoint').getAttribute('waypoint')).toBe('a');
  expect(result).toContain('<!-- keep -->');
  expect(doc.querySelectorAll('Sequence > *')).toHaveLength(2);
});
test('clearing a field restores the runtime default and XML values are escaped', () => {
  const changed = updateParameter(xml,'2','waypoint','A&B');
  expect(changed).toContain('A&amp;B');
  expect(updateParameter(changed,'2','waypoint','')).not.toContain('waypoint=');
  expect(() => updateParameter(xml,'1','ID','Other')).toThrow('Only task parameters');
  expect(() => parameterNodes('<broken>')).toThrow('Invalid waypoint task');
});
