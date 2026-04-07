
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

static half3 restoreVideoColor(half3 rgb, float colorBoost) {
    float3 c = float3(rgb);
    float boost = clamp(colorBoost, 0.80f, 1.40f);
    float t = (boost - 1.0f) / 0.40f;

    // Recover contrast/saturation from AVPlayer->BGRA conversion path that can appear washed out.
    c = pow(clamp(c, 0.0f, 1.0f), float3(1.08f + 0.08f * t));
    float luma = dot(c, float3(0.2126f, 0.7152f, 0.0722f));
    c = mix(float3(luma), c, 1.08f + 0.12f * t);
    c = (c - 0.5f) * (1.06f + 0.10f * t) + 0.5f;
    c = min(c * (1.00f + 0.03f * t), 1.0f);

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

    // Expand usable disparity range and keep the middle depth near zero parallax for comfort.
    depth = smoothstep(0.08f, 0.92f, depth);
    depth = pow(depth, 0.82f);

    float centeredDepth = depth - 0.5f;
    float deadZone = 0.035f;
    if (abs(centeredDepth) < deadZone) {
        centeredDepth = 0.0f;
    } else {
        centeredDepth = sign(centeredDepth) * ((abs(centeredDepth) - deadZone) / (0.5f - deadZone)) * 0.5f;
    }
    float eyeSign = (eyeIndex == 0) ? -1.0f : 1.0f;
    // Horizontal UV shift is the core 2D->3D conversion step.
    float shift = centeredDepth * separation * 0.70f;
    shift = clamp(shift, -0.0032f, 0.0032f);
    uv.x -= eyeSign * shift;

    uv = clamp(uv, 0.0f, 1.0f);
    half4 src = colorTex.sample(s, uv);
    half3 converted = restoreVideoColor(src.rgb, colorBoost);
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
    depth = smoothstep(0.08f, 0.92f, depth);
    depth = pow(depth, 0.82f);

    float centeredDepth = depth - 0.5f;
    float deadZone = 0.035f;
    if (abs(centeredDepth) < deadZone) {
        centeredDepth = 0.0f;
    } else {
        centeredDepth = sign(centeredDepth) * ((abs(centeredDepth) - deadZone) / (0.5f - deadZone)) * 0.5f;
    }
    float eyeSign = (eyeIndex == 0) ? -1.0f : 1.0f;
    float shift = centeredDepth * depthMultiplier * 0.70f;
    shift = clamp(shift, -0.0032f, 0.0032f);
    uv.x -= eyeSign * shift;

    uv = clamp(uv, 0.0f, 1.0f);
    half4 src = colorTex.sample(s, uv);
    half3 converted = restoreVideoColor(src.rgb, colorBoost);
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
