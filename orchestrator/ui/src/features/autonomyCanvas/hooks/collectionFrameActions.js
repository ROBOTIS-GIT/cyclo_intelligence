import { collectionRequest } from "../../../utils/collectionApi";
export default function collectionFrameActions(session) {
  const { episode, selectedFrame, latest, datasetGeneration, operation, setEpisode,
    setSelectedFrame, setError, setReasoningContextVersion, refreshList, refreshReasonPresets } = session;
  const deleteFrames = async (frameIds) => {
    let failure;
    const ok = await operation(async () => {
      try {
        const current = latest.current.episode;
        if (!current?.ended_at) throw new Error("Save the episode before deleting frames.");
        datasetGeneration.current += 1;
        const {episode: updated, warning} = await collectionRequest(`episodes/${current.id}/frames/delete`, {frame_ids: frameIds});
        datasetGeneration.current += 1;
        if (warning) setError(warning);
        latest.current.episode = updated;
        setEpisode(updated);
        setSelectedFrame(previous => {
          if (!previous) return null;
          const surviving = updated.observations.find(frame => frame.frame_id === previous.frame_id);
          const oldIndex = current.observations.findIndex(frame => frame.frame_id === previous.frame_id);
          const nextIds = new Set(current.observations.slice(Math.max(0, oldIndex)).map(frame => frame.frame_id));
          const next = surviving || updated.observations.find(frame => nextIds.has(frame.frame_id)) || updated.observations.at(-1);
          return next ? {...next, labelRevision: updated.annotations.find(label => label.frame_ids[0] === next.frame_id)?.revision || 0} : null;
        });
      } catch (err) { failure = err; throw err; }
      // Deletion is already committed; a list refresh failure must not invite a duplicate delete.
      try { await Promise.all([refreshList(), refreshReasonPresets()]); }
      catch (err) { setError(`Frames deleted. Could not refresh the episode list: ${err.message}`); }
    });
    if (!ok) throw failure || new Error("Another dataset operation is in progress. Try again shortly.");
    return true;
  };
  const annotate = (fields) =>
    operation(async () => {
      if (!episode || !selectedFrame) throw new Error("Select a saved frame first");
      const saved = await collectionRequest(`episodes/${episode.id}/annotations`, {
        ...fields,
        frame_id: selectedFrame.frame_id,
        expected_revision: selectedFrame.labelRevision || 0,
      });
      setEpisode(saved);
      setSelectedFrame((previous) => ({
        ...previous,
        ...saved.observations.find((frame) => frame.frame_id === previous.frame_id),
        labelRevision: saved.annotations.find((a) => a.frame_ids[0] === selectedFrame.frame_id)
          .revision,
      }));
      await Promise.all([refreshList(), refreshReasonPresets()]);
    });
  const instruction = (text) =>
    operation(async () => {
      await collectionRequest(`episodes/${episode.id}/instructions`, { text });
      setReasoningContextVersion((value) => value + 1);
    });
  return { deleteFrames, annotate, instruction,
    selectFrame: frame => setSelectedFrame(frame ? {
      ...frame,
      labelRevision: latest.current.episode?.annotations.find(label => label.frame_ids[0] === frame.frame_id)?.revision || 0,
    } : null),
  };
}
