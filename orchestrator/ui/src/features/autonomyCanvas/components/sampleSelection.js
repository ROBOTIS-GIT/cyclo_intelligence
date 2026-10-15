export const USER_SAMPLE_SCHEMA = "user_count_v1";

export function currentSampleSelection(sample) {
  return sample?.schema_version === USER_SAMPLE_SCHEMA ? sample : null;
}
