# LoadStore Test

Offscreen Vulkan app that measures render-pass load/store cost on tiling GPUs.
Each pass is a full-screen triangle running a dependent FMA loop (ALU iterations
adjustable), into one color attachment. loadOp / storeOp are selectable:
loadOp LOAD / CLEAR / DONT_CARE / NONE, storeOp STORE / DONT_CARE / NONE
(NONE needs VK_EXT/KHR_load_store_op_none; the app reports if it is missing).

Build: push to GitHub, run the "Build APK" workflow, download the
`loadstore-test-apk` artifact. Shaders are precompiled (shader.vert/.frag ->
*_spv.h via `glslangValidator -V -S <stage> f -o f.spv && xxd -i f.spv`).

Buttons: Run once; ALU sweep (chosen load/store vs DONT_CARE/DONT_CARE,
with delta); matrix of all loadOp x storeOp combos.
