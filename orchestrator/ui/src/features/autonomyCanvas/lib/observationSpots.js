export function observationSpots(configuration) {
  const referenced = new Set((configuration?.plan?.observations || []).map(p => p.spot_id));
  return (configuration?.spots || []).filter(s => s.metadata?.role === "observation" || referenced.has(s.id));
}

