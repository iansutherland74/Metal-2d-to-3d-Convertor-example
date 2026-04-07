# 2D to 3D Video Converter for Apple Vision Pro

This project is a streamlined visionOS app focused on one task: converting monoscopic 2D video into stereo 3D playback in an ImmersiveSpace using Metal and CompositorServices.

## What it does

- Plays a source 2D video through `AVPlayerItemVideoOutput`.
- Generates a depth map per frame using a bundled Core ML depth model.
- Falls back to a luminance-based depth estimate if model depth is unavailable.
- Converts each frame to stereo by applying per-eye depth-based parallax in a Metal fragment shader.

## Current focus

This repository has been reduced from the original spatial rendering sample to keep only the conversion pipeline needed for 2D-to-3D video playback:

- Kept: video decode, depth inference/fallback, stereo conversion shader, and immersive render loop.
- Removed: environment sphere rendering, portal/passthrough controls, and hand-gesture interaction plumbing.

## Rendering behavior

- On Apple Vision Pro, the compositor uses layered rendering where available.
- On the Simulator, the app can use dedicated layout.
- Stereo views are rendered in a single pass when the runtime supports vertex amplification.