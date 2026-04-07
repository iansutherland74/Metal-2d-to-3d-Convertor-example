# Vision UI -> Metal Port Plan

## Scope reality
Vision UI is a React/Next.js component system. A direct source-to-source conversion to Metal is not possible because the platforms, runtime, and rendering models differ.

This project now contains a Metal-native first phase that ports the core visual language into shader-driven rendering.

## What is already ported
- Panel chrome moved to Metal fragment stage.
- Rounded panel masking in shader.
- Glass/tint treatment in shader.
- Animated ornament ring effect in shader.
- Runtime tuning controls exposed in SwiftUI and fed into Metal constants each frame.

## Parameter bridge (SwiftUI -> ObjC++ -> Metal)
- panelGlassAmount
- panelEdgeGlow
- panelCornerRadius
- panelOrnamentIntensity
- panelOrnamentSpeed
- panelTintStrength

## Vision UI concept mapping
- Card / glass surfaces -> panel fragment styling + rounded signed-distance mask.
- Ornament components -> animated radial ring and pulse terms.
- Motion/react transitions -> time-based shader animation and scene-time modulation.
- Environment styling -> existing environment HDR and portal controls.
- Component tokens -> SRConfiguration properties + fragment constants.

## Recommended next phases
1. Extract a reusable panel renderer abstraction.
2. Add text/icon atlas support for label and control rendering in Metal.
3. Build interaction hit regions in world space and dispatch events to a small scene graph.
4. Add multi-panel layout primitives (stack, ornament anchor, dock).
5. Add theme assets and token files (JSON) loaded at startup.

## Notes
- Keep this as a reinterpretation, not a byte-for-byte clone.
- Preserve visionOS interaction comfort defaults while adding style options.
