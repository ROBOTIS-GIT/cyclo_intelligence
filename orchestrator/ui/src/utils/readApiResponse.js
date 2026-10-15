// Preserve HTTP failures when a proxy returns an HTML error page.
export default async function readApiResponse(response, fallback="Request failed") {
  const status=response.status;
  const failure=message=>Object.assign(new Error(message),{status});
  let data;
  try {data=await response.json();}
  catch(error) {
    if(error.name==="AbortError")throw error;
    const reason=status===504 ? "Server response timed out. Check the current operation before retrying."
      : [502,503].includes(status) ? "Server temporarily unavailable. Please retry shortly."
      : "Server returned an invalid response. Please retry.";
    throw failure(`${reason} (HTTP ${status})`);
  }
  if(!response.ok)throw failure(typeof data?.detail==="string"?data.detail:`${fallback} (HTTP ${status})`);
  return data;
}
