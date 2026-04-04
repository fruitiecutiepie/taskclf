import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("taskclfPayloadChooser", {
  ok: (version: string) => {
    ipcRenderer.send("taskclf-payload-chooser-submit", version);
  },
  cancel: () => {
    ipcRenderer.send("taskclf-payload-chooser-cancel");
  },
});
