import { contextBridge, ipcRenderer } from "electron";

contextBridge.exposeInMainWorld("taskclfUpdateChooser", {
  submit: (payload: {
    launcher: boolean;
    core: boolean;
    coreVersion: string | null;
  }) => {
    ipcRenderer.send("taskclf-update-chooser-submit", payload);
  },
  cancel: () => {
    ipcRenderer.send("taskclf-update-chooser-cancel");
  },
});
