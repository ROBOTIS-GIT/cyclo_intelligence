import { readBtNodeCatalog } from "./btCatalogClient";
import { getBtNodeReadiness } from "../features/mapWorkspace/lib/btNodeApi";

jest.mock("../features/mapWorkspace/lib/btNodeApi", () => ({ getBtNodeReadiness: jest.fn() }));

beforeEach(() => { jest.resetAllMocks(); });

test("waits for ROS registration before requesting the catalog", async () => {
  let register;
  getBtNodeReadiness.mockResolvedValueOnce({ ready: false })
    .mockImplementationOnce(() => new Promise(resolve => { register = resolve; }));
  const call = jest.fn().mockResolvedValue({ success: true, catalog_json: "[]" });
  const pending = readBtNodeCatalog(call, { attempts: 2, intervalMs: 0 });
  await new Promise(resolve => setTimeout(resolve, 5));
  expect(call).not.toHaveBeenCalled();
  register({ ready: true });
  expect(await pending).toEqual({ success: true, catalog_json: "[]" });
  expect(call).toHaveBeenCalledTimes(1);
  expect(call).toHaveBeenCalledWith("/bt/nodes/catalog", "interfaces/srv/GetNodeCatalog", {});
});

test("does not probe an absent catalog when readiness times out", async () => {
  getBtNodeReadiness.mockResolvedValue({ ready: false });
  const call = jest.fn();
  await expect(readBtNodeCatalog(call, { attempts: 2, intervalMs: 0 })).rejects.toThrow("not ready");
  expect(call).not.toHaveBeenCalled();
});

test("does not bypass an unavailable readiness API", async () => {
  getBtNodeReadiness.mockRejectedValue(new Error("ROS discovery unavailable"));
  const call = jest.fn();
  await expect(readBtNodeCatalog(call)).rejects.toThrow("ROS discovery unavailable");
  expect(call).not.toHaveBeenCalled();
});

test("propagates a service failure after discovery without claiming success", async () => {
  getBtNodeReadiness.mockResolvedValue({ ready: true });
  const call = jest.fn().mockRejectedValue(new Error("service disappeared"));
  await expect(readBtNodeCatalog(call)).rejects.toThrow("service disappeared");
});
