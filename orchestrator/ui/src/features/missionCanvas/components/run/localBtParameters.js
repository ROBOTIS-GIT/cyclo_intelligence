// Edit attributes only: keep tree structure, IDs, comments and other subtrees intact.
export function parameterNodes(xml, catalog = []) {
  const doc = new DOMParser().parseFromString(xml, 'text/xml');
  if (doc.querySelector('parsererror')) throw new Error('Invalid waypoint task XML');
  const elements = [...doc.querySelectorAll('BehaviorTree *')];
  return elements.map((el, index) => {
    const type = el.tagName === 'Action' || el.tagName === 'Control' ? el.getAttribute('ID') : el.tagName;
    const entry = catalog.find(item => item.tag === type);
    const defaults = Object.fromEntries((entry?.ports || []).map(p => [p.name, p.default]));
    const attrs = Object.fromEntries([...el.attributes].filter(a => !['name', 'ID', 'bt_collapsed'].includes(a.name)).map(a => [a.name, a.value]));
    return { id: String(index), data: { nodeType: type, label: el.getAttribute('name') || type, params: { ...defaults, ...attrs } } };
  }).filter(n => Object.keys(n.data.params).length);
}

export function updateParameter(xml, id, key, value) {
  if (['name', 'ID', 'bt_collapsed'].includes(key)) throw new Error('Only task parameters can be edited');
  const doc = new DOMParser().parseFromString(xml, 'text/xml');
  if (doc.querySelector('parsererror')) throw new Error('Invalid waypoint task XML');
  const element = [...doc.querySelectorAll('BehaviorTree *')][Number(id)];
  if (!element) throw new Error('Task node no longer exists');
  if (String(value).trim() === '') element.removeAttribute(key);
  else element.setAttribute(key, String(value));
  return new XMLSerializer().serializeToString(doc);
}
