import { collectionRequest } from "../../../utils/collectionApi";
export default function collectionEpisodeActions(session, stopRecording) {
  const { latest, pending, active, operation, setEpisode, setSelectedFrame,
    setCaptureReceipt, refreshList, refreshReasonPresets } = session;
  const { enabled, mapName } = latest.current;
  const start = (fields) =>
    operation(async () => {
      if (!enabled || active) return;
      if (!mapName) throw new Error("Load a map before starting an episode.");
      const data = await collectionRequest("episodes", { ...fields, map_name: mapName || null });
      latest.current.episode = data;
      setEpisode(data);
      setSelectedFrame(null);
      setCaptureReceipt("");
      await refreshList();
    });
  const load = (id) => {
    // Selecting the current episode is not a reload: keep the frame/editor mounted.
    // Refresh has its own action and preserves the selected frame.
    if (id && latest.current.episode?.id === id) return Promise.resolve(true);
    return operation(async () => {
      if (latest.current.episode && !latest.current.episode.ended_at) return;
      const data = id ? await collectionRequest(`episodes/${id}`) : null;
      latest.current.episode = data;
      setEpisode(data);
      setSelectedFrame(null);
      setCaptureReceipt("");
    });
  };
  const discardEpisode = async () => {
    stopRecording();
    // A capture already in flight must finish before its episode is removed.
    while (pending.current) await new Promise((resolve) => setTimeout(resolve, 25));
    return operation(async () => {
      const current = latest.current.episode;
      if (!current) return;
      if (!current.ended_at) {
        const ended = await collectionRequest(`episodes/${current.id}/end`, {});
        setEpisode(ended);
      }
      await collectionRequest(`episodes/${current.id}`, undefined, "DELETE");
      setEpisode(null);
      setSelectedFrame(null);
      setCaptureReceipt("Episode deleted.");
      await Promise.all([refreshList(), refreshReasonPresets()]);
    });
  };
  const deleteEpisode = async (id) => {
    if (!id) return false;
    if (latest.current.episode?.id === id) return discardEpisode();
    return operation(async () => {
      await collectionRequest(`episodes/${id}`, undefined, "DELETE");
      await Promise.all([refreshList(), refreshReasonPresets()]);
    });
  };
  const saveEpisode = async ({ allowEmpty = false } = {}) => {
    stopRecording();
    // Finish an in-flight capture before ending the episode; do not request another frame.
    while (pending.current) await new Promise((resolve) => setTimeout(resolve, 25));
    return operation(async () => {
      const current = latest.current.episode;
      if (!allowEmpty && !current?.observations?.length)
        throw new Error("Capture at least one frame before saving the episode.");
      const data = await collectionRequest(`episodes/${current.id}/end`, {});
      setEpisode(data);
      setSelectedFrame(null);
      await refreshList();
    });
  };
  const startReturnEpisode = async ({ keepRecording = false } = {}) => {
    if (!keepRecording) stopRecording();
    const sourceId = latest.current.episode?.id;
    while (pending.current) await new Promise(resolve => setTimeout(resolve, 25));
    let child = null;
    const ok = await operation(async () => {
      const current = latest.current;
      if (!sourceId || current.episode?.id !== sourceId || !current.enabled || current.mapName !== current.episode.map_name)
        throw new Error("The recording session changed before the return segment could be saved.");
      child = await collectionRequest(`episodes/${sourceId}/return`, {});
      // Publish immediately as well as through React: startRecording may follow in the same turn.
      latest.current.episode = child;
      setEpisode(child); setSelectedFrame(null); setCaptureReceipt("Return capture ready.");
      await refreshList();
    });
    return ok ? child : null;
  };
  const startSegment = async (fields) => {
    const sourceId = latest.current.episode?.id;
    while (pending.current) await new Promise(resolve => setTimeout(resolve, 25));
    let child = null;
    const ok = await operation(async () => {
      const current = latest.current;
      if (!sourceId || current.episode?.id !== sourceId || !current.enabled || current.episode.ended_at)
        throw new Error("The recording session changed before saving the next segment.");
      child = await collectionRequest(`episodes/${sourceId}/segment`, { ...fields, map_name: current.mapName });
      latest.current.episode = child;
      setEpisode(child); setSelectedFrame(null); setCaptureReceipt("");
      await refreshList();
    });
    return ok ? child : null;
  };
  return { start, load, discardEpisode, deleteEpisode, finish: saveEpisode,
    startReturnEpisode, startSegment };
}
