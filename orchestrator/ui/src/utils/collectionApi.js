import readApiResponse from "./readApiResponse";
// Copyright 2026 ROBOTIS CO., LTD.
// Licensed under the Apache License, Version 2.0.
export async function collectionRequest(path, body, method) {
  const response = await fetch(`/api/collection/${path}`, {
    ...(method ? { method } : {}),
    ...(body === undefined
      ? {}
      : {
          method: method || "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }),
  });
  return readApiResponse(response, "Collection request failed");
}
