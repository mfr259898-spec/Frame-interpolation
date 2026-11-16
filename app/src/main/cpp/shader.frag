#version 450
layout(push_constant) uniform PC { int iters; float seed; } pc;
layout(location = 0) out vec4 outColor;
void main() {
    // Long dependent FMA chain. Trip count is a runtime push constant,
    // so the compiler can neither unroll it away nor remove it.
    float a = gl_FragCoord.x * 0.001 + pc.seed;
    float b = gl_FragCoord.y * 0.001 + 0.5;
    for (int i = 0; i < pc.iters; i++) {
        a = fma(a, 0.99991, b);
        b = fma(b, 0.99983, a * 0.0001);
    }
    outColor = vec4(a, b, a * b, 1.0);
}
