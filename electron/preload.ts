import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("electronHost", {
  invoke(command: unknown): Promise<void> {
    return ipcRenderer.invoke("taskclf-host", command);
  },
});
