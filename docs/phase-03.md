# NEXORA AI — Phase 03: 3D Neural Forest & AI Avatar Engine Architecture

## 1. Overview

Phase 03 establishes the signature visual identity of **NEXORA AI**: **CYBER-NATURE INTELLIGENCE**. 

It introduces a custom 3D Perspective Projection Engine rendering an interactive, bioluminescent **Neural Forest Topology** alongside a procedural 3D **Holographic AI Avatar**. Both engines bind directly to the existing Riverpod state management without introducing external platform-native binary overhead.

```
                              [ AI CORE (0,0,0) ]
                                 /     │     \
                                /      │      \
           [ LLM Engine ] ◄────┼───────┼───────┼────► [ RAG Retrieval ]
           [ Vision / 3D ] ◄───┼───────┼───────┼────► [ Speech / Voice ]
           [ FastMCP ] ◄───────┼───────┼───────┼────► [ Multi-Agent ]
                               \       │       /
                                \      │      /
                              [ ROOTS: Data / DB / Redis ]
```

---

## 2. 3D Engine Architecture & Projection Mathematics

### 3D Perspective Projection Model
Rather than relying on heavy native C++ wrappers that break across web/mobile platforms, the rendering engine uses direct 3D matrix transformation and perspective projection math:

Given a 3D point $P = (X, Y, Z)$ rotated by angles $\theta_x$ (pitch) and $\theta_y$ (yaw):
1. **Rotation around Y-axis (Yaw)**:
   $$X_1 = X \cos\theta_y + Z \sin\theta_y$$
   $$Z_1 = -X \sin\theta_y + Z \cos\theta_y$$
2. **Rotation around X-axis (Pitch)**:
   $$Y_2 = Y \cos\theta_x - Z_1 \sin\theta_x$$
   $$Z_2 = Y \sin\theta_x + Z_1 \cos\theta_x$$
3. **Perspective Projection**:
   Given focal length $f = 400$ and camera Z-offset $Z_{\text{camera}} = 500$:
   $$\text{Scale Factor } S = \frac{f}{Z_2 + Z_{\text{camera}}} \cdot \text{scale}$$
   $$X_{\text{screen}} = X_{\text{center}} + X_1 \cdot S$$
   $$Y_{\text{screen}} = Y_{\text{center}} + Y_2 \cdot S$$

### Depth Sorting (Painter's Algorithm)
All projected nodes are sorted by $Z_2$ depth before painting to guarantee correct visual occlusion (furthest nodes and branches drawn first, foreground nodes and glow highlights drawn last).

---

## 3. Bioluminescent State Animations

Animations react directly to Riverpod `AIState` (`idle`, `listening`, `thinking`, `processing`, `speaking`, `success`, `error`):

- **`idle`**: Slow breathing sine-wave pulse ($0.8\times \leftrightarrow 1.2\times$).
- **`listening`**: Cyan energy waves and enlarged glow radii ($1.5\times$).
- **`thinking`**: Violet node activation with accelerated particle streams along branches.
- **`processing`**: Bright cyan particles travelling along neural lines.
- **`speaking`**: Holographic avatar mouth line renders dynamic voice waveform oscillations.
- **`success`**: Emerald green particle burst glow.
- **`error`**: Controlled red warning glow highlights.

---

## 4. 3D Holographic AI Avatar Engine

The 3D Holographic Avatar (`AvatarView`) renders:
- **Head Wireframe Oval & Outer Holographic Energy Ring**: Rotates 360 degrees around the head center.
- **Glowing Eye Orbitals & Pupils**: React to state colors.
- **Voice Waveform Lip Sync Interface**: Renders dynamic bezier/sine wave oscillations during `AIState.speaking`, creating the foundation for future WebRTC audio amplitude binding.

---

## 5. Performance Strategy (60 FPS Goal)

1. **Isolated Repaint Drivers**: `NeuralForestVisualizer` and `AvatarView` use `SingleTickerProviderStateMixin` with `AnimatedBuilder` to restrict animation frame repaints strictly to the `CustomPaint` canvas.
2. **Zero Dependency Overhead**: Avoids native plugin bridge context switches, achieving 60 FPS across Web, Android, iOS, Windows, macOS, and Linux.
3. **Hit Box Scaling**: Tap hit-testing scales radius dynamically based on $S$ (`28.0 * scaleFactor`), ensuring easy node selection regardless of 3D depth or rotation.

---

## 6. How to Run & Verify

1. Navigate to Flutter app:
   ```bash
   cd mobile/flutter_app
   ```
2. Run Flutter app:
   ```bash
   flutter run -d chrome
   ```
3. Test Node Selection: Tap on `LLM Engine`, `RAG Retrieval`, `Vision & 3D-CNN`, `FastMCP`, or `Multi-Agent` to inspect the glassmorphic `NodeInfoPanel`.
4. Test 3D Camera Controls: Drag to rotate the 3D Neural Forest in 360 degrees; pinch to zoom in/out; tap the reset icon to restore origin camera view.
5. Test AI State Animations: Click choice chips (`LISTENING`, `THINKING`, `PROCESSING`, `SPEAKING`, `SUCCESS`, `ERROR`) to observe reactive 3D tree and avatar state animations.
