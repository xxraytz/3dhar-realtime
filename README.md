# Real-Time Skeleton Activity Recognition

A Python desktop application that recognizes human activities from **Kinect v2 skeletons for up to six people**, with a **PyQt6 GUI**. It processes a live sensor stream or a recorded scene and displays activity labels for each tracked person. Optional person identification uses enrolled motion profiles.

## GUI

- Color video or skeleton view with activity and identity labels.
- Pause and resume capture.
- Switch views through **Settings → Video mode**.
- Select enrolled people through **Settings → Choose people** when identification is enabled.

Capture, rendering and recognition run in separate Qt threads to keep the interface responsive.

## Model

The bundled **ResNetXV2** is a residual CNN with Squeeze-and-Excitation channel attention, loaded with **TensorFlow/Keras** from `data/nn-model/data003/` in **SavedModel** format.

For each person, the application normalizes 17 skeleton joints over a 32-frame window. Distances to 32 reference poses form a `32 × 32` activity map used as the network input.

The model predicts **eight activities plus `no_action`**:

`standing`, `walking`, `sitting`, `falling`, `grasp`, `ungrasp`, `laying`, `post-fall-laying`, `no_action`.

## Kinect setup

You need **Windows x64**, a **Kinect v2** with its powered PC adapter, a compatible **USB 3.0** port, **Python 3.8.10 x64** and **Git**.

1. Disconnect the sensor and install [Kinect for Windows SDK 2.0](https://www.microsoft.com/en-us/download/details.aspx?id=44561).
2. Connect the sensor to its adapter, connect power, then plug the adapter into USB 3.0.
3. Wait for driver installation and check color and body tracking with the Kinect SDK tools.

## Install and run

In PowerShell, from the repository root:

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-desktop.txt
python service.py --source kinect
```

This opens the GUI and starts activity recognition. Model weights and the default configuration (`config.example.json`) are included; no training is required to start.

To use a custom configuration, copy the example, edit it and run:

```powershell
Copy-Item config.example.json config.json
python service.py --source kinect --config config.json
```

To replay a recorded scene:

```powershell
python service.py --source recording --recording "path_to_scene"
```

Replace `path_to_scene` with the path to a scene directory containing synchronized `Body/FileSkeleton.txt` and `Color/person_stream.mp4` files. Replay supports one recorded person; live Kinect capture supports up to six.

## License

[MIT](LICENSE) — Copyright (c) 2026 Egor Surkov.
