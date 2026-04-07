import SwiftUI
import RealityKit

struct ConverterControlsView: View {
    @Environment(\.openImmersiveSpace) var openImmersiveSpace
    @Environment(\.dismissImmersiveSpace) var dismissImmersiveSpace

    @State private var showImmersiveSpace = false
    @State private var hasAutoStartedImmersiveSpace = false
    @State private var panelOffsetX = 0.0
    @State private var panelOffsetY = 0.0
    @State private var panelOffsetZ = -1.0
    @State private var panelScale = 1.0
    @State private var disparityStrength = 3.0
    @State private var colorBoost = 1.40
    @State private var playbackScrubTime = 0.0
    @State private var isScrubbingPlayback = false

    // Direct bridge object to native renderer settings.
    private let rendererConfiguration: Video3DConfiguration
    @ObservedObject private var playbackController: VideoPlaybackController

    init(_ rendererConfig: Video3DConfiguration, playbackController: VideoPlaybackController) {
        rendererConfiguration = rendererConfig
        self.playbackController = playbackController
        _panelOffsetX = State(initialValue: rendererConfig.panelOffsetX)
        _panelOffsetY = State(initialValue: rendererConfig.panelOffsetY)
        _panelOffsetZ = State(initialValue: rendererConfig.panelOffsetZ)
        _panelScale = State(initialValue: rendererConfig.panelScale)
        _disparityStrength = State(initialValue: rendererConfig.disparityStrength)
        _colorBoost = State(initialValue: rendererConfig.colorBoost)
        _playbackScrubTime = State(initialValue: playbackController.currentTime)
    }

    var body: some View {
        NavigationStack {
            Form {
                sessionSection
                playbackSection
                panelSection
                stereoSection
                colorSection
            }
            .navigationTitle("Spatial Video Controls")
        }
        .task {
            // Auto-open immersive rendering once so the conversion surface appears immediately.
            if !hasAutoStartedImmersiveSpace {
                hasAutoStartedImmersiveSpace = true
                rendererConfiguration.panelOffsetX = 0.0
                rendererConfiguration.panelOffsetY = 0.0
                rendererConfiguration.panelOffsetZ = -1.0
                rendererConfiguration.panelScale = 1.0
                let result = await openImmersiveSpace(id: "ImmersiveSpace")
                if case .opened = result {
                    showImmersiveSpace = true
                }
            }
        }
        .task {
            // Keep control values in sync with native state that may change outside SwiftUI.
            while true {
                try? await Task.sleep(nanoseconds: 16_666_667) // ~60fps
                panelOffsetX = rendererConfiguration.panelOffsetX
                panelOffsetY = rendererConfiguration.panelOffsetY
                panelOffsetZ = rendererConfiguration.panelOffsetZ
                if !isScrubbingPlayback {
                    playbackScrubTime = playbackController.currentTime
                }
            }
        }
        .onChange(of: showImmersiveSpace) { _, newValue in
            Task {
                if newValue {
                    rendererConfiguration.panelOffsetX = 0.0
                    rendererConfiguration.panelOffsetY = 0.0
                    rendererConfiguration.panelOffsetZ = -1.0
                    rendererConfiguration.panelScale = 1.0
                    await openImmersiveSpace(id: "ImmersiveSpace")
                } else {
                    await dismissImmersiveSpace()
                }
            }
        }
        .onChange(of: panelOffsetX) { _, newValue in
            rendererConfiguration.panelOffsetX = newValue
        }
        .onChange(of: panelOffsetY) { _, newValue in
            rendererConfiguration.panelOffsetY = newValue
        }
        .onChange(of: panelOffsetZ) { _, newValue in
            rendererConfiguration.panelOffsetZ = newValue
        }
        .onChange(of: panelScale) { _, newValue in
            rendererConfiguration.panelScale = newValue
        }
        .onChange(of: disparityStrength) { _, newValue in
            rendererConfiguration.disparityStrength = newValue
        }
        .onChange(of: colorBoost) { _, newValue in
            rendererConfiguration.colorBoost = newValue
        }
    }

    private func requestSeek(to requestedTime: Double) {
        // Clamp seek requests to valid media bounds before forwarding to AVPlayer.
        let clampedTime = min(max(0.0, requestedTime), max(playbackController.duration, 0.0))
        playbackScrubTime = clampedTime
        playbackController.seek(to: clampedTime)
    }

    private var sessionSection: some View {
        Section("Session") {
            Toggle(showImmersiveSpace ? "Immersive Playing" : "Start Immersive Playback", isOn: $showImmersiveSpace)
                .toggleStyle(.switch)
        }
    }

    private var playbackSection: some View {
        Section("Playback") {
            HStack {
                Button("-10s") { playbackController.skip(by: -10.0) }
                Button(playbackController.isPlaying ? "Pause" : "Play") { playbackController.togglePlayback() }
                Button("+10s") { playbackController.skip(by: 10.0) }
            }
            .buttonStyle(.bordered)

            VStack(alignment: .leading, spacing: 6) {
                Text("\(formatTime(playbackController.currentTime)) / \(formatTime(playbackController.duration))")
                    .monospacedDigit()
                Slider(value: $playbackScrubTime,
                       in: 0...max(playbackController.duration, 0.1),
                       onEditingChanged: handlePlaybackScrub)
            }
        }
    }

    private var panelSection: some View {
        Section("Panel") {
            VStack(alignment: .leading, spacing: 6) {
                Text("Position X: \(panelOffsetX, specifier: "%.2f") m")
                Slider(value: $panelOffsetX, in: -2.5...2.5)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Position Y: \(panelOffsetY, specifier: "%.2f") m")
                Slider(value: $panelOffsetY, in: 0.3...2.2)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Distance: \(panelOffsetZ, specifier: "%.2f") m")
                Slider(value: $panelOffsetZ, in: -4.0...(-0.6))
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Scale: \(panelScale, specifier: "%.2f")x")
                Slider(value: $panelScale, in: 0.5...2.0)
            }
        }
    }

    private var stereoSection: some View {
        Section("Stereo") {
            VStack(alignment: .leading, spacing: 6) {
                Text("3D Strength: \(disparityStrength, specifier: "%.1f")")
                    .monospacedDigit()
                Slider(value: $disparityStrength, in: 0...13, step: 0.1)
            }

            HStack {
                Button("Off") { disparityStrength = 0 }
                Button("Comfort") { disparityStrength = 2.0 }
                Button("Default") { disparityStrength = 3.0 }
                Button("Strong") { disparityStrength = 5.0 }
            }
            .buttonStyle(.bordered)
        }
    }

    private var colorSection: some View {
        Section("Color") {
            VStack(alignment: .leading, spacing: 6) {
                Text("Color Boost: \(colorBoost, specifier: "%.2f")x")
                    .monospacedDigit()
                Slider(value: $colorBoost, in: 0.80...1.40, step: 0.01)
            }

            HStack {
                Button("Natural") { colorBoost = 1.00 }
                Button("Boost") { colorBoost = 1.12 }
                Button("Vivid") { colorBoost = 1.24 }
            }
            .buttonStyle(.bordered)
        }
    }

    private func handlePlaybackScrub(_ isEditing: Bool) {
        isScrubbingPlayback = isEditing
        if !isEditing {
            requestSeek(to: playbackScrubTime)
        }
    }

    private func formatTime(_ seconds: Double) -> String {
        guard seconds.isFinite, seconds > 0 else {
            return "00:00"
        }

        let totalSeconds = Int(seconds.rounded(.down))
        let minutes = totalSeconds / 60
        let remainingSeconds = totalSeconds % 60
        return String(format: "%02d:%02d", minutes, remainingSeconds)
    }
}
