export const jobRunning = job => ["queued", "training", "running", "cancelling"].includes(job?.status);
