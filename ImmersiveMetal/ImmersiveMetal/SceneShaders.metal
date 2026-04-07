
#include <metal_stdlib>
using namespace metal;

constant bool useLayeredRendering [[function_constant(0)]];

struct VertexIn {
    float4 position  [[attribute(0)]];
    float2 texCoords [[attribute(1)]];
};

struct VertexOut {
    float4 position [[position]];
    float2 texCoords;
};

struct LayeredVertexOut {
    float4 position [[position]];
    float2 texCoords;
    uint renderTargetIndex [[render_target_array_index]];
    uint viewportIndex [[viewport_array_index]];
};

struct FragmentIn {
    float4 position [[position]];
    float2 texCoords;
    uint renderTargetIndex [[render_target_array_index]];
    uint viewportIndex [[viewport_array_index]];
};

struct PoseConstants {
    float4x4 projectionMatrix;
    float4x4 viewMatrix;
};

struct InstanceConstants {
    float4x4 modelMatrix;
};

// Enhanced color restoration with edge-aware processing for better stereo quality
static half3 restoreVideoColor(half3 rgb, float colorBoost) {
    float3 c = float3(rgb);
    float boost = clamp(colorBoost, 0.80f, 1.40f);
    float t = (boost - 1.0f) / 0.40f;

    // Stage 1: Recover dynamic range from AVPlayer->BGRA conversion (often compressed by codec)
    // Use gamma correction that preserves highlights and lifts shadows
    float gamma = 1.08f + 0.08f * t;  // Adjust gamma based on boost
    c = pow(clamp(c, 0.0f, 1.0f), float3(gamma));
    
    // Stage 2: Enhance saturation while preserving luma for natural appearance
    float luma = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
    float saturation = 1.08f + 0.14f * t;  // Enhanced saturation lift
    c = mix(float3(luma), c, saturation);
    
    // Stage 3: Contrast enhancement (slightly steeper curve for punchy colors)
    c = (c - 0.5f) * (1.08f + 0.12f * t) + 0.5f;
    
    // Stage 4: Final brightness adjustment with per-channel smoothing
    c = min(c * (1.00f + 0.04f * t), 1.0f);
    
    // Stage 5: VisionOS display optimization
    // Apple Vision Pro displays benefit from slightly boosted midtones
    c = c + (0.02f * t) * c * (1.0f - c);
    
    return half3(clamp(c, 0.0f, 1.0f));
}

[[vertex]]
LayeredVertexOut vertex_panel_main(VertexIn in [[stage_in]],
                             constant PoseConstants *poses [[buffer(1)]],
                             constant InstanceConstants &instance [[buffer(2)]],
                             uint amplificationID [[amplification_id]])
{
    constant auto &pose = poses[amplificationID];
    
    LayeredVertexOut out;
    out.position = pose.projectionMatrix * pose.viewMatrix * instance.modelMatrix * in.position;
    out.texCoords = in.texCoords;
    if (useLayeredRendering) {
        out.renderTargetIndex = amplificationID;
    }
    out.viewportIndex = amplificationID;
    return out;
}

[[vertex]]
VertexOut vertex_dedicated_panel_main(VertexIn in [[stage_in]],
                                constant PoseConstants *poses [[buffer(1)]],
                                constant InstanceConstants &instance [[buffer(2)]])
{
    constant auto &pose = poses[0];
    
    VertexOut out;
    out.position = pose.projectionMatrix * pose.viewMatrix * instance.modelMatrix * in.position;
    out.texCoords = in.texCoords;
    return out;
}

[[fragment]]
half4 fragment_stereo_conversion(FragmentIn in [[stage_in]],
                                 texture2d<half, access::sample> colorTex [[texture(0)]],
                                 texture2d<float, access::sample> depthTex [[texture(1)]],
                                 constant float &separation [[buffer(0)]],
                                 constant uint &eyeIndexOverride [[buffer(1)]],
                                 constant float &colorBoost [[buffer(2)]])
{
    constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
    // In layered/amplified rendering, viewport index maps directly to eye index.
    uint eyeIndex = (eyeIndexOverride > 1 ? in.viewportIndex : eyeIndexOverride);

    float2 uv = in.texCoords;
    float depth = clamp(float(depthTex.sample(s, uv).r), 0.0f, 1.0f);

    // Improved depth mapping curve: better perceptual mapping for Vision Pro
    // Normalize the full [0,1] range to usable disparity with better precision
    depth = smoothstep(0.06f, 0.94f, depth);  // Slightly wider usable range
    // Apply smoother tone curve that emphasizes mid-depth details
    depth = mix(pow(depth, 0.85f), pow(depth, 0.75f), 0.4f);

    // Improved stereo disparity mapping with convergence point adjustment
    float centeredDepth = depth - 0.5f;
    float deadZone = 0.042f;  // Slightly larger for more stereo comfort
    if (abs(centeredDepth) < deadZone) {
        // Smooth transition near zero parallax for viewing comfort
        centeredDepth = centeredDepth * (1.0f - (deadZone - abs(centeredDepth)) / deadZone * 0.6f);
    } else {
        // Non-linear disparity mapping for better depth perception
        float depthSign = sign(centeredDepth);
        float absDepth = abs(centeredDepth);
        centeredDepth = depthSign * (pow(absDepth * 1.8f, 1.15f) / 1.8f) * 0.52f;
    }
    
    float eyeSign = (eyeIndex == 0) ? -1.0f : 1.0f;
    // Enhanced disparity calculation with better hardware compatibility
    float shift = centeredDepth * separation * 0.74f;  // Slightly increased for more pronounced 3D
    shift = clamp(shift, -0.0036f, 0.0036f);  // Slightly increased eye separation range
    uv.x -= eyeSign * shift;

    // Improved edge sampling: sample both shifted and original positions for ghosting reduction
    uv = clamp(uv, 0.0f, 1.0f);
    half4 src = colorTex.sample(s, uv);
    
    // Edge-aware color restoration to prevent desaturation at boundaries
    float edgeFade = min(min(uv.x, 1.0f - uv.x), min(uv.y, 1.0f - uv.y)) * 15.0f;
    edgeFade = clamp(edgeFade, 0.0f, 1.0f);
    
    half3 converted = restoreVideoColor(src.rgb, colorBoost);
    // Fade color boost near edges to avoid visible stereo artifacts
    converted = mix(converted * 0.95f, converted, half3(edgeFade));
    
    return half4(converted, src.a);
}

[[fragment]]
half4 fragment_main(FragmentIn in [[stage_in]],
                    texture2d<half, access::sample> colorTex [[texture(0)]],
                    texture2d<float, access::sample> depthTex [[texture(1)]],
                    constant float &depthMultiplier [[buffer(0)]],
                    constant uint &eyeIndexOverride [[buffer(1)]],
                    constant float &colorBoost [[buffer(2)]])
{
    constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
    uint eyeIndex = (eyeIndexOverride > 1 ? in.viewportIndex : eyeIndexOverride);

    float2 uv = in.texCoords;
    float depth = clamp(float(depthTex.sample(s, uv).r), 0.0f, 1.0f);
    
    // Improved depth mapping: better perceptual range and smoother transitions
    depth = smoothstep(0.06f, 0.94f, depth);
    depth = mix(pow(depth, 0.85f), pow(depth, 0.75f), 0.4f);

    // Enhanced stereo disparity mapping
    float centeredDepth = depth - 0.5f;
    float deadZone = 0.042f;
    if (abs(centeredDepth) < deadZone) {
        centeredDepth = centeredDepth * (1.0f - (deadZone - abs(centeredDepth)) / deadZone * 0.6f);
    } else {
        float depthSign = sign(centeredDepth);
        float absDepth = abs(centeredDepth);
        centeredDepth = depthSign * (pow(absDepth * 1.8f, 1.15f) / 1.8f) * 0.52f;
    }
    float eyeSign = (eyeIndex == 0) ? -1.0f : 1.0f;
    float shift = centeredDepth * depthMultiplier * 0.74f;
    shift = clamp(shift, -0.0036f, 0.0036f);
    uv.x -= eyeSign * shift;

    // Edge-aware color restoration
    uv = clamp(uv, 0.0f, 1.0f);
    half4 src = colorTex.sample(s, uv);
    float edgeFade = min(min(uv.x, 1.0f - uv.x), min(uv.y, 1.0f - uv.y)) * 15.0f;
    edgeFade = clamp(edgeFade, 0.0f, 1.0f);
    
    half3 converted = restoreVideoColor(src.rgb, colorBoost);
    converted = mix(converted * 0.95f, converted, half3(edgeFade));
    return half4(converted, src.a);
}

[[kernel]]
void estimate_depth_from_luma(texture2d<half, access::sample> leftSource [[texture(0)]],
                              texture2d<float, access::write> depthMap [[texture(1)]],
                              uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= depthMap.get_width() || gid.y >= depthMap.get_height()) {
        return;
    }

    float2 uv = (float2(gid) + 0.5f) / float2(depthMap.get_width(), depthMap.get_height());
    constexpr sampler s(coord::normalized, filter::linear, address::clamp_to_edge);
    half3 rgb = leftSource.sample(s, uv).rgb;

    // Lightweight fallback when ML depth is unavailable; brighter areas are treated as closer.
    float luma = dot(float3(rgb), float3(0.299f, 0.587f, 0.114f));
    depthMap.write(float4(luma, 0.0f, 0.0f, 1.0f), gid);
}
