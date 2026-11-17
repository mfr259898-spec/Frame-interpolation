#version 450
// VK_QCOM_render_pass_shader_resolve: last subpass reads every sample of the
// multisampled attachment (still in tile memory) and writes one resolved color.
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInputMS msaaIn;
layout(push_constant) uniform PC { int samples; } pc;
layout(location = 0) out vec4 outColor;
void main() {
    vec4 sum = vec4(0.0);
    for (int i = 0; i < pc.samples; i++) sum += subpassLoad(msaaIn, i);
    outColor = sum / float(pc.samples);
}
