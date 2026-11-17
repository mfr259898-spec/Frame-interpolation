// Tile load/store overhead test: offscreen full-screen passes with a
// runtime-adjustable ALU load, and selectable loadOp / storeOp (incl. NONE).
#include <jni.h>
#include <vulkan/vulkan.h>
#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "vert_spv.h"
#include "frag_spv.h"
#include "resolve_spv.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "loadstore", __VA_ARGS__)

// VK_EXT_load_store_op_none enum values (defined locally so old headers work)
static const int LOAD_OP_NONE_V = 1000400000;
static const int STORE_OP_NONE_V = 1000301000;

struct Ctx {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue q = VK_NULL_HANDLE;
    uint32_t qf = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mp{};
    bool noneExt = false;
    bool shaderResolveExt = false;
    std::string info;
} G;

static std::string g_err;

static double nowMs() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static bool initVk() {
    if (G.dev) return true;
    VkApplicationInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    ai.pApplicationName = "loadstore";
    ai.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &ai;
    if (vkCreateInstance(&ici, nullptr, &G.inst) != VK_SUCCESS) { g_err = "vkCreateInstance failed"; return false; }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(G.inst, &n, nullptr);
    if (!n) { g_err = "no Vulkan physical device"; return false; }
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(G.inst, &n, pds.data());
    G.pd = pds[0];
    vkGetPhysicalDeviceMemoryProperties(G.pd, &G.mp);

    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(G.pd, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qp(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(G.pd, &qn, qp.data());
    bool found = false;
    for (uint32_t i = 0; i < qn; i++)
        if (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { G.qf = i; found = true; break; }
    if (!found) { g_err = "no graphics queue"; return false; }

    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(G.pd, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(G.pd, nullptr, &en, exts.data());
    const char* noneName = nullptr;
    const char* resolveName = nullptr;
    for (auto& e : exts) {
        if (!strcmp(e.extensionName, "VK_EXT_load_store_op_none")) { noneName = "VK_EXT_load_store_op_none"; }
        else if (!strcmp(e.extensionName, "VK_KHR_load_store_op_none") && !noneName) noneName = "VK_KHR_load_store_op_none";
        if (!strcmp(e.extensionName, "VK_QCOM_render_pass_shader_resolve")) resolveName = "VK_QCOM_render_pass_shader_resolve";
    }
    G.noneExt = noneName != nullptr;
    G.shaderResolveExt = resolveName != nullptr;
    std::vector<const char*> enabled;
    if (noneName) enabled.push_back(noneName);
    if (resolveName) enabled.push_back(resolveName);

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = G.qf;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)enabled.size();
    dci.ppEnabledExtensionNames = enabled.data();
    if (vkCreateDevice(G.pd, &dci, nullptr, &G.dev) != VK_SUCCESS) { g_err = "vkCreateDevice failed"; return false; }
    vkGetDeviceQueue(G.dev, G.qf, 0, &G.q);

    VkCommandPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = G.qf;
    if (vkCreateCommandPool(G.dev, &pci, nullptr, &G.pool) != VK_SUCCESS) { g_err = "vkCreateCommandPool failed"; return false; }

    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(G.pd, &pp);
    char buf[640];
    snprintf(buf, sizeof(buf), "%s\nVulkan %u.%u.%u  driver 0x%x\nload/store op NONE ext: %s\nQCOM shader resolve ext: %s",
             pp.deviceName, VK_VERSION_MAJOR(pp.apiVersion), VK_VERSION_MINOR(pp.apiVersion),
             VK_VERSION_PATCH(pp.apiVersion), pp.driverVersion, noneName ? noneName : "NOT supported",
             resolveName ? "yes" : "NOT supported");
    G.info = buf;
    return true;
}

static int findMem(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < G.mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (G.mp.memoryTypes[i].propertyFlags & want) == want) return (int)i;
    return -1;
}

struct Run {
    VkImage img = VK_NULL_HANDLE, img2 = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE, mem2 = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE, view2 = VK_NULL_HANDLE;
    VkRenderPass rp = VK_NULL_HANDLE;
    VkFramebuffer fb = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSet dset = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE, pl2 = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE, pipe2 = VK_NULL_HANDLE;
    VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE, rfs = VK_NULL_HANDLE;
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    ~Run() {
        vkDeviceWaitIdle(G.dev);
        if (fence) vkDestroyFence(G.dev, fence, nullptr);
        if (cb) vkFreeCommandBuffers(G.dev, G.pool, 1, &cb);
        if (pipe2) vkDestroyPipeline(G.dev, pipe2, nullptr);
        if (pipe) vkDestroyPipeline(G.dev, pipe, nullptr);
        if (pl2) vkDestroyPipelineLayout(G.dev, pl2, nullptr);
        if (pl) vkDestroyPipelineLayout(G.dev, pl, nullptr);
        if (dpool) vkDestroyDescriptorPool(G.dev, dpool, nullptr);
        if (dsl) vkDestroyDescriptorSetLayout(G.dev, dsl, nullptr);
        if (rfs) vkDestroyShaderModule(G.dev, rfs, nullptr);
        if (fs) vkDestroyShaderModule(G.dev, fs, nullptr);
        if (vs) vkDestroyShaderModule(G.dev, vs, nullptr);
        if (fb) vkDestroyFramebuffer(G.dev, fb, nullptr);
        if (rp) vkDestroyRenderPass(G.dev, rp, nullptr);
        if (view2) vkDestroyImageView(G.dev, view2, nullptr);
        if (view) vkDestroyImageView(G.dev, view, nullptr);
        if (img2) vkDestroyImage(G.dev, img2, nullptr);
        if (img) vkDestroyImage(G.dev, img, nullptr);
        if (mem2) vkFreeMemory(G.dev, mem2, nullptr);
        if (mem) vkFreeMemory(G.dev, mem, nullptr);
    }
};

struct Result { bool ok = false; double msPerPass = 0; int passesPerSubmit = 0; double footprintMB = 0; };

#define VKCHK(x, msg) do { if ((x) != VK_SUCCESS) { g_err = msg; return res; } } while (0)

static bool makeImage(VkFormat fmt, uint32_t W, uint32_t H, int S, VkImageUsageFlags usage,
                      VkImage& img, VkDeviceMemory& mem, VkImageView& view) {
    VkImageCreateInfo ic{};
    ic.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ic.imageType = VK_IMAGE_TYPE_2D;
    ic.format = fmt;
    ic.extent = {W, H, 1};
    ic.mipLevels = 1;
    ic.arrayLayers = 1;
    ic.samples = (VkSampleCountFlagBits)S;
    ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = usage;
    ic.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (vkCreateImage(G.dev, &ic, nullptr, &img) != VK_SUCCESS) { g_err = "vkCreateImage failed"; return false; }
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(G.dev, img, &mr);
    int mt = findMem(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mt < 0) { g_err = "no device-local memory type"; return false; }
    VkMemoryAllocateInfo mai{};
    mai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    mai.allocationSize = mr.size;
    mai.memoryTypeIndex = mt;
    if (vkAllocateMemory(G.dev, &mai, nullptr, &mem) != VK_SUCCESS) { g_err = "vkAllocateMemory failed (image too big?)"; return false; }
    if (vkBindImageMemory(G.dev, img, mem, 0) != VK_SUCCESS) { g_err = "bind failed"; return false; }
    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    if (vkCreateImageView(G.dev, &vi, nullptr, &view) != VK_SUCCESS) { g_err = "view failed"; return false; }
    return true;
}

// storeSel: 0 STORE, 1 DONT_CARE, 2 NONE, 3 RESOLVE (fixed-function resolve attachment),
//           4 RESOLVE_QCOM (VK_QCOM_render_pass_shader_resolve, 2nd subpass)
static Result runTest(int loadSel, int storeSel, int fmtSel, int sampSel, int resSel, int iters, int draws, double seconds) {
    Result res;
    if (!initVk()) return res;

    static const int loadOps[] = {VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_CLEAR,
                                  VK_ATTACHMENT_LOAD_OP_DONT_CARE, LOAD_OP_NONE_V};
    static const int storeOps[] = {VK_ATTACHMENT_STORE_OP_STORE, VK_ATTACHMENT_STORE_OP_DONT_CARE, STORE_OP_NONE_V,
                                   VK_ATTACHMENT_STORE_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE};
    static const VkFormat fmts[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT};
    static const int bpp[] = {4, 8, 16};
    static const int samps[] = {1, 2, 4};
    static const uint32_t dims[][2] = {{1280, 720}, {1920, 1080}, {2560, 1440}, {3840, 2160}};

    if ((loadSel == 3 || storeSel == 2) && !G.noneExt) { g_err = "NONE load/store op needs VK_EXT/KHR_load_store_op_none (not supported)"; return res; }
    const bool resolveStd = storeSel == 3, resolveQcom = storeSel == 4, resolving = resolveStd || resolveQcom;
    if (resolveQcom && !G.shaderResolveExt) { g_err = "VK_QCOM_render_pass_shader_resolve not supported by this driver"; return res; }

    const VkFormat fmt = fmts[fmtSel];
    const int S = samps[sampSel];
    const uint32_t W = dims[resSel][0], H = dims[resSel][1];
    if (resolving && S < 2) { g_err = "resolve store modes need MSAA 2x or 4x"; return res; }
    draws = std::max(1, draws);

    VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (resolveQcom) usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
    VkImageFormatProperties ifp;
    if (vkGetPhysicalDeviceImageFormatProperties(G.pd, fmt, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
                                                 usage, 0, &ifp) != VK_SUCCESS ||
        !(ifp.sampleCounts & S) || ifp.maxExtent.width < W || ifp.maxExtent.height < H) {
        g_err = "format/samples/resolution/usage not supported";
        return res;
    }

    Run r;
    if (!makeImage(fmt, W, H, S, usage, r.img, r.mem, r.view)) return res;
    if (resolving && !makeImage(fmt, W, H, 1, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, r.img2, r.mem2, r.view2)) return res;

    // Attachment 0: (MSAA) color. Attachment 1 (resolve modes): single-sample resolve target.
    VkAttachmentDescription ad[2]{};
    ad[0].format = fmt;
    ad[0].samples = (VkSampleCountFlagBits)S;
    ad[0].loadOp = (VkAttachmentLoadOp)loadOps[loadSel];
    ad[0].storeOp = (VkAttachmentStoreOp)storeOps[storeSel];
    ad[0].stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ad[0].stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    ad[0].initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ad[0].finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    ad[1] = ad[0];
    ad[1].samples = VK_SAMPLE_COUNT_1_BIT;
    ad[1].loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    ad[1].storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkAttachmentReference msaaRef{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference resRef{1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkAttachmentReference inRef{0, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkSubpassDescription sp[2]{};
    sp[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sp[0].colorAttachmentCount = 1;
    sp[0].pColorAttachments = &msaaRef;
    if (resolveStd) sp[0].pResolveAttachments = &resRef;
    uint32_t subpassCount = 1;
    if (resolveQcom) {
        sp[1].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        sp[1].flags = 0x4 | 0x8;  // FRAGMENT_REGION_BIT_QCOM | SHADER_RESOLVE_BIT_QCOM
        sp[1].inputAttachmentCount = 1;
        sp[1].pInputAttachments = &inRef;
        sp[1].colorAttachmentCount = 1;
        sp[1].pColorAttachments = &resRef;
        subpassCount = 2;
    }
    // Serializes consecutive passes on the same images (same for every op combo).
    VkSubpassDependency dep[3]{};
    dep[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dep[0].dstSubpass = 0;
    dep[0].srcStageMask = dep[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dep[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dep[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    uint32_t depCount = 1;
    if (resolveQcom) {
        dep[1].srcSubpass = 0;
        dep[1].dstSubpass = 1;
        dep[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dep[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        dep[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        dep[1].dstAccessMask = VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
        dep[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
        dep[2] = dep[0];
        dep[2].dstSubpass = 1;
        depCount = 3;
    }
    VkRenderPassCreateInfo rpi{};
    rpi.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpi.attachmentCount = resolving ? 2 : 1;
    rpi.pAttachments = ad;
    rpi.subpassCount = subpassCount;
    rpi.pSubpasses = sp;
    rpi.dependencyCount = depCount;
    rpi.pDependencies = dep;
    VKCHK(vkCreateRenderPass(G.dev, &rpi, nullptr, &r.rp), "vkCreateRenderPass failed");

    VkImageView fbViews[2] = {r.view, r.view2};
    VkFramebufferCreateInfo fi{};
    fi.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    fi.renderPass = r.rp;
    fi.attachmentCount = resolving ? 2 : 1;
    fi.pAttachments = fbViews;
    fi.width = W;
    fi.height = H;
    fi.layers = 1;
    VKCHK(vkCreateFramebuffer(G.dev, &fi, nullptr, &r.fb), "framebuffer failed");

    VkShaderModuleCreateInfo smi{};
    smi.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    std::vector<uint32_t> vcode((vert_spv_len + 3) / 4), fcode((frag_spv_len + 3) / 4), rcode((resolve_spv_len + 3) / 4);
    memcpy(vcode.data(), vert_spv, vert_spv_len);
    memcpy(fcode.data(), frag_spv, frag_spv_len);
    memcpy(rcode.data(), resolve_spv, resolve_spv_len);
    smi.codeSize = vert_spv_len;
    smi.pCode = vcode.data();
    VKCHK(vkCreateShaderModule(G.dev, &smi, nullptr, &r.vs), "vs failed");
    smi.codeSize = frag_spv_len;
    smi.pCode = fcode.data();
    VKCHK(vkCreateShaderModule(G.dev, &smi, nullptr, &r.fs), "fs failed");
    smi.codeSize = resolve_spv_len;
    smi.pCode = rcode.data();
    VKCHK(vkCreateShaderModule(G.dev, &smi, nullptr, &r.rfs), "resolve fs failed");

    VkPushConstantRange pcr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8};
    VkPipelineLayoutCreateInfo pli{};
    pli.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VKCHK(vkCreatePipelineLayout(G.dev, &pli, nullptr, &r.pl), "pipeline layout failed");

    if (resolveQcom) {
        VkDescriptorSetLayoutBinding lb{0, VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo dli{};
        dli.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        dli.bindingCount = 1;
        dli.pBindings = &lb;
        VKCHK(vkCreateDescriptorSetLayout(G.dev, &dli, nullptr, &r.dsl), "desc set layout failed");
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1};
        VkDescriptorPoolCreateInfo dpi{};
        dpi.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpi.maxSets = 1;
        dpi.poolSizeCount = 1;
        dpi.pPoolSizes = &ps;
        VKCHK(vkCreateDescriptorPool(G.dev, &dpi, nullptr, &r.dpool), "desc pool failed");
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = r.dpool;
        dai.descriptorSetCount = 1;
        dai.pSetLayouts = &r.dsl;
        VKCHK(vkAllocateDescriptorSets(G.dev, &dai, &r.dset), "desc set alloc failed");
        VkDescriptorImageInfo dii{VK_NULL_HANDLE, r.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkWriteDescriptorSet wds{};
        wds.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wds.dstSet = r.dset;
        wds.descriptorCount = 1;
        wds.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        wds.pImageInfo = &dii;
        vkUpdateDescriptorSets(G.dev, 1, &wds, 0, nullptr);
        VkPipelineLayoutCreateInfo pli2 = pli;
        pli2.setLayoutCount = 1;
        pli2.pSetLayouts = &r.dsl;
        VKCHK(vkCreatePipelineLayout(G.dev, &pli2, nullptr, &r.pl2), "pipeline layout 2 failed");
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = r.vs;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = r.fs;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vin{};
    vin.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkViewport vp{0, 0, (float)W, (float)H, 0, 1};
    VkRect2D sc{{0, 0}, {W, H}};
    VkPipelineViewportStateCreateInfo vps{};
    vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vps.viewportCount = 1;
    vps.pViewports = &vp;
    vps.scissorCount = 1;
    vps.pScissors = &sc;
    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = (VkSampleCountFlagBits)S;
    VkPipelineColorBlendAttachmentState cba{};
    cba.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cbs{};
    cbs.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cbs.attachmentCount = 1;
    cbs.pAttachments = &cba;
    VkGraphicsPipelineCreateInfo gpi{};
    gpi.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpi.stageCount = 2;
    gpi.pStages = stages;
    gpi.pVertexInputState = &vin;
    gpi.pInputAssemblyState = &ia;
    gpi.pViewportState = &vps;
    gpi.pRasterizationState = &rs;
    gpi.pMultisampleState = &ms;
    gpi.pColorBlendState = &cbs;
    gpi.layout = r.pl;
    gpi.renderPass = r.rp;
    gpi.subpass = 0;
    VKCHK(vkCreateGraphicsPipelines(G.dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &r.pipe), "pipeline failed");

    if (resolveQcom) {
        // Shader-resolve subpass pipeline: rasterizationSamples == input attachment samples,
        // output to the single-sample resolve target, no sample shading.
        stages[1].module = r.rfs;
        gpi.layout = r.pl2;
        gpi.subpass = 1;
        VKCHK(vkCreateGraphicsPipelines(G.dev, VK_NULL_HANDLE, 1, &gpi, nullptr, &r.pipe2),
              "shader-resolve pipeline failed (driver rejected the QCOM resolve setup)");
    }

    VkCommandBufferAllocateInfo cai{};
    cai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cai.commandPool = G.pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VKCHK(vkAllocateCommandBuffers(G.dev, &cai, &r.cb), "alloc cb failed");
    VkFenceCreateInfo fci{};
    fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VKCHK(vkCreateFence(G.dev, &fci, nullptr, &r.fence), "fence failed");

    auto submit = [&]() -> bool {
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &r.cb;
        if (vkQueueSubmit(G.q, 1, &si, r.fence) != VK_SUCCESS) return false;
        if (vkWaitForFences(G.dev, 1, &r.fence, VK_TRUE, 10000000000ULL) != VK_SUCCESS) return false;
        vkResetFences(G.dev, 1, &r.fence);
        return true;
    };
    auto begin = [&]() {
        vkResetCommandBuffer(r.cb, 0);
        VkCommandBufferBeginInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(r.cb, &bi);
    };

    // One-time layout transition UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
    begin();
    VkImageMemoryBarrier b[2]{};
    for (int i = 0; i < 2; i++) {
        b[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        b[i].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT;
        b[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b[i].newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    }
    b[0].image = r.img;
    b[1].image = r.img2;
    vkCmdPipelineBarrier(r.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0, 0, nullptr, 0, nullptr, resolving ? 2 : 1, b);
    vkEndCommandBuffer(r.cb);
    if (!submit()) { g_err = "layout transition submit failed"; return res; }

    VkClearValue cv[2]{};
    cv[0].color.float32[3] = 1.0f;
    cv[1].color.float32[3] = 1.0f;
    auto record = [&](int passes) {
        begin();
        for (int i = 0; i < passes; i++) {
            VkRenderPassBeginInfo rb{};
            rb.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            rb.renderPass = r.rp;
            rb.framebuffer = r.fb;
            rb.renderArea = {{0, 0}, {W, H}};
            rb.clearValueCount = resolving ? 2 : 1;
            rb.pClearValues = cv;
            vkCmdBeginRenderPass(r.cb, &rb, VK_SUBPASS_CONTENTS_INLINE);
            vkCmdBindPipeline(r.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r.pipe);
            for (int d = 0; d < draws; d++) {  // overlapping full-screen draws, same pass
                struct { int32_t iters; float seed; } pc = {iters, 0.37f * (float)d};
                vkCmdPushConstants(r.cb, r.pl, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8, &pc);
                vkCmdDraw(r.cb, 3, 1, 0, 0);
            }
            if (resolveQcom) {
                vkCmdNextSubpass(r.cb, VK_SUBPASS_CONTENTS_INLINE);
                vkCmdBindPipeline(r.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r.pipe2);
                vkCmdBindDescriptorSets(r.cb, VK_PIPELINE_BIND_POINT_GRAPHICS, r.pl2, 0, 1, &r.dset, 0, nullptr);
                struct { int32_t samples; float pad; } pc2 = {S, 0.0f};
                vkCmdPushConstants(r.cb, r.pl2, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8, &pc2);
                vkCmdDraw(r.cb, 3, 1, 0, 0);
            }
            vkCmdEndRenderPass(r.cb);
        }
        vkEndCommandBuffer(r.cb);
    };

    // Calibrate: pick passes-per-submit so one submit lasts ~20 ms (1..64 passes).
    record(1);
    if (!submit()) { g_err = "submit failed"; return res; }
    double t0 = nowMs();
    for (int i = 0; i < 3; i++) if (!submit()) { g_err = "submit failed"; return res; }
    double t1 = (nowMs() - t0) / 3.0;
    int N = std::max(1, std::min(64, (int)(20.0 / std::max(t1, 0.01) + 0.5)));
    record(N);
    for (int i = 0; i < 2; i++) if (!submit()) { g_err = "submit failed"; return res; }

    int submits = 0;
    double start = nowMs(), el = 0;
    do {
        if (!submit()) { g_err = "submit failed (GPU timeout/hang?)"; return res; }
        submits++;
        el = nowMs() - start;
    } while (el < seconds * 1000.0 || submits < 3);

    res.ok = true;
    res.msPerPass = el / ((double)submits * N);
    res.passesPerSubmit = N;
    res.footprintMB = (double)W * H * bpp[fmtSel] * S / 1048576.0;
    return res;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_loadstore_MainActivity_nativeInfo(JNIEnv* env, jobject) {
    if (!initVk()) return env->NewStringUTF(("Vulkan init failed: " + g_err).c_str());
    return env->NewStringUTF(G.info.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_loadstore_MainActivity_nativeLastError(JNIEnv* env, jobject) {
    return env->NewStringUTF(g_err.c_str());
}

// returns {msPerPass, passesPerSubmit, footprintMB} or null on error
extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_example_loadstore_MainActivity_nativeRun(JNIEnv* env, jobject, jint load, jint store, jint fmt,
                                                  jint samples, jint resolution, jint iters, jint draws, jdouble seconds) {
    g_err.clear();
    Result r = runTest(load, store, fmt, samples, resolution, iters, draws, seconds);
    if (!r.ok) return nullptr;
    jdouble out[3] = {r.msPerPass, (double)r.passesPerSubmit, r.footprintMB};
    jdoubleArray arr = env->NewDoubleArray(3);
    env->SetDoubleArrayRegion(arr, 0, 3, out);
    return arr;
}
