export default async function request(path, body, signal) {
  const response = await fetch(`/api/collection/${path}`, { signal, ...(body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }) });
  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail)
      ? data.detail.map(item => `${(item.loc || []).filter(key => key !== "body").join(".")}: ${item.msg}`).join("; ")
      : data.detail;
    throw new Error(typeof detail === "string" ? detail : "Segmentation request failed");
  }
  return data;
}
