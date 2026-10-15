// Server-owned episodes. Browser drafts are never imported as training data.
import { useCallback, useEffect } from "react";
import { collectionRequest } from "../../../utils/collectionApi";
import useCollectionSession from "./useCollectionSession";
import useEpisodeCapture from "./useEpisodeCapture";
import collectionEpisodeActions from "./collectionEpisodeActions";
import collectionFrameActions from "./collectionFrameActions";

// Public facade for Record and Train; the session remains the only mutation owner.
export default function useCollectDraft({ enabled, mapName, context = {}, getExecutionContext }) {
  const session = useCollectionSession({ enabled, mapName, context, getExecutionContext });
  const capture = useEpisodeCapture(session);
  const { episode, episodes, reasonPresets, selectedFrame, error, busy, captureReceipt,
    reasoningContextVersion, active, latest, operation, setEpisode, refresh, refreshList } = session;
  const { stopRecording } = capture;
  const end = useCallback(
    () =>
      operation(async () => {
        const id = latest.current.episode?.id;
        if (!id) return;
        setEpisode(await collectionRequest(`episodes/${id}/end`, {}));
        await refreshList();
      }),
    [operation, refreshList, latest, setEpisode],
  );
  useEffect(() => {
    if (!enabled || episode?.map_name !== mapName) stopRecording();
    if (active && (!enabled || episode.map_name !== mapName) && !busy) end();
  }, [active, enabled, mapName, episode?.map_name, end, busy, stopRecording]);

  return { episode, episodes, reasonPresets, selectedFrame, error, busy, captureReceipt,
    reasoningContextVersion, active, end, refresh,
    refreshEpisodes: () => operation(refreshList),
    ...capture,
    ...collectionEpisodeActions(session, stopRecording),
    ...collectionFrameActions(session),
  };
}
