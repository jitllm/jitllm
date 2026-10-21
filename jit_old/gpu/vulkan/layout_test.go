package vulkan

import (
	"testing"
	"unsafe"
)

// TestStructLayout pins every C struct offset against vulkan_core.h.
//
// These layouts are transcribed by hand because there is no cgo to check them,
// and a wrong offset neither fails to compile nor panics: the driver misreads a
// field and returns a plausible error or corrupts memory.
//
// The numbers below are offsetof() from the C header, not from this file.
func TestStructLayout(t *testing.T) {
	type field struct {
		name string
		got  uintptr
		want uintptr
	}
	check := func(what string, size, wantSize uintptr, fs ...field) {
		if size != wantSize {
			t.Errorf("%s: size %d, C says %d", what, size, wantSize)
		}
		for _, f := range fs {
			if f.got != f.want {
				t.Errorf("%s.%s: offset %d, C says %d", what, f.name, f.got, f.want)
			}
		}
	}
	var (
		pp  physProps
		mp2 memProps2
		mb  memBudgetEXT
		xp  extProps
		ai  appInfo
		ici instanceCI
		qci queueCI
		dci deviceCI
		df  dotFeat
		mp  memProps
		bci bufferCI
		ma  memAlloc
		db  descBinding
		dl  descLayoutCI
		pl  pipeLayoutCI
		sm  shaderCI
		st  stageCI
		cp  computeCI
		dp  descPoolCI
		da  descSetAI
		wd  writeDesc
		cpc cmdPoolCI
		cba cmdBufAI
		cbb cmdBeginI
		si  submitInfo
		pp2 physProps2
		sgp subgroupProps
		scp sizeCtlProps
		scf sizeCtlFeat
		rsc reqSizeCI
		idp idProps
	)
	o := func(p, f unsafe.Pointer) uintptr { return uintptr(f) - uintptr(p) }
	check("VkApplicationInfo", unsafe.Sizeof(ai), 48,
		field{"pNext", o(unsafe.Pointer(&ai), unsafe.Pointer(&ai.pNext)), 8},
		field{"pApplicationName", o(unsafe.Pointer(&ai), unsafe.Pointer(&ai.pName)), 16},
		field{"applicationVersion", o(unsafe.Pointer(&ai), unsafe.Pointer(&ai.appVer)), 24},
		field{"pEngineName", o(unsafe.Pointer(&ai), unsafe.Pointer(&ai.pEngine)), 32},
		field{"engineVersion", o(unsafe.Pointer(&ai), unsafe.Pointer(&ai.engVer)), 40},
		field{"apiVersion", o(unsafe.Pointer(&ai), unsafe.Pointer(&ai.apiVer)), 44})
	check("VkInstanceCreateInfo", unsafe.Sizeof(ici), 64,
		field{"pApplicationInfo", o(unsafe.Pointer(&ici), unsafe.Pointer(&ici.pApp)), 24},
		field{"ppEnabledExtensionNames", o(unsafe.Pointer(&ici), unsafe.Pointer(&ici.pExts)), 56})
	check("VkDeviceQueueCreateInfo", unsafe.Sizeof(qci), 40,
		field{"queueFamilyIndex", o(unsafe.Pointer(&qci), unsafe.Pointer(&qci.family)), 20},
		field{"pQueuePriorities", o(unsafe.Pointer(&qci), unsafe.Pointer(&qci.prio)), 32})
	check("VkDeviceCreateInfo", unsafe.Sizeof(dci), 72,
		field{"pQueueCreateInfos", o(unsafe.Pointer(&dci), unsafe.Pointer(&dci.pQueues)), 24},
		field{"ppEnabledExtensionNames", o(unsafe.Pointer(&dci), unsafe.Pointer(&dci.pExts)), 56},
		field{"pEnabledFeatures", o(unsafe.Pointer(&dci), unsafe.Pointer(&dci.pFeats)), 64})
	check("VkPhysicalDeviceShaderIntegerDotProductFeatures", unsafe.Sizeof(df), 24,
		field{"shaderIntegerDotProduct", o(unsafe.Pointer(&df), unsafe.Pointer(&df.on)), 16})
	check("VkPhysicalDeviceMemoryProperties", unsafe.Sizeof(mp), 520,
		field{"memoryTypes", o(unsafe.Pointer(&mp), unsafe.Pointer(&mp.types)), 4},
		field{"memoryHeapCount", o(unsafe.Pointer(&mp), unsafe.Pointer(&mp.heapCount)), 260},
		field{"memoryHeaps", o(unsafe.Pointer(&mp), unsafe.Pointer(&mp.heaps)), 264})
	// VkPhysicalDeviceProperties. deviceType at 16 is what device.go ranks on;
	// reading 20 would yield the first bytes of deviceName, which devTypeRank
	// maps to "CPU, refuse".
	check("VkPhysicalDeviceProperties", unsafe.Sizeof(pp), 824,
		field{"driverVersion", o(unsafe.Pointer(&pp), unsafe.Pointer(&pp.driverVer)), 4},
		field{"vendorID", o(unsafe.Pointer(&pp), unsafe.Pointer(&pp.vendorID)), 8},
		field{"deviceID", o(unsafe.Pointer(&pp), unsafe.Pointer(&pp.deviceID)), 12},
		field{"deviceType", o(unsafe.Pointer(&pp), unsafe.Pointer(&pp.devType)), 16},
		field{"deviceName", o(unsafe.Pointer(&pp), unsafe.Pointer(&pp.name)), 20},
		field{"pipelineCacheUUID", o(unsafe.Pointer(&pp), unsafe.Pointer(&pp.cacheUUID)), 276},
		field{"limits", o(unsafe.Pointer(&pp), unsafe.Pointer(&pp.limits)), 296},
		field{"sparseProperties", o(unsafe.Pointer(&pp), unsafe.Pointer(&pp.sparse)), 800})
	if unsafe.Alignof(pp) != 8 {
		// limits holds VkDeviceSize members, so C aligns the whole struct to 8
		// and the driver stores 64-bit values into it. A Go struct of uint32
		// and byte arrays would align to 4.
		t.Errorf("VkPhysicalDeviceProperties: align %d, C says 8", unsafe.Alignof(pp))
	}
	check("VkPhysicalDeviceMemoryProperties2", unsafe.Sizeof(mp2), 536,
		field{"pNext", o(unsafe.Pointer(&mp2), unsafe.Pointer(&mp2.pNext)), 8},
		field{"memoryProperties", o(unsafe.Pointer(&mp2), unsafe.Pointer(&mp2.props)), 16})
	check("VkPhysicalDeviceMemoryBudgetPropertiesEXT", unsafe.Sizeof(mb), 272,
		field{"pNext", o(unsafe.Pointer(&mb), unsafe.Pointer(&mb.pNext)), 8},
		field{"heapBudget", o(unsafe.Pointer(&mb), unsafe.Pointer(&mb.budget)), 16},
		field{"heapUsage", o(unsafe.Pointer(&mb), unsafe.Pointer(&mb.usage)), 144})
	check("VkExtensionProperties", unsafe.Sizeof(xp), 260,
		field{"specVersion", o(unsafe.Pointer(&xp), unsafe.Pointer(&xp.specVer)), 256})
	// deviceUUID and driverUUID are adjacent sixteen-byte arrays: reading 32
	// instead of 16 gives a well-formed UUID of the driver, which would merge
	// two real cards into one.
	check("VkPhysicalDeviceIDProperties", unsafe.Sizeof(idp), 64,
		field{"pNext", o(unsafe.Pointer(&idp), unsafe.Pointer(&idp.pNext)), 8},
		field{"deviceUUID", o(unsafe.Pointer(&idp), unsafe.Pointer(&idp.deviceUUID)), 16},
		field{"driverUUID", o(unsafe.Pointer(&idp), unsafe.Pointer(&idp.driverUUID)), 32},
		field{"deviceLUID", o(unsafe.Pointer(&idp), unsafe.Pointer(&idp.deviceLUID)), 48},
		field{"deviceNodeMask", o(unsafe.Pointer(&idp), unsafe.Pointer(&idp.nodeMask)), 56},
		field{"deviceLUIDValid", o(unsafe.Pointer(&idp), unsafe.Pointer(&idp.luidValid)), 60})
	check("VkBufferCreateInfo", unsafe.Sizeof(bci), 56,
		field{"size", o(unsafe.Pointer(&bci), unsafe.Pointer(&bci.size)), 24},
		field{"usage", o(unsafe.Pointer(&bci), unsafe.Pointer(&bci.usage)), 32},
		field{"pQueueFamilyIndices", o(unsafe.Pointer(&bci), unsafe.Pointer(&bci.pFamily)), 48})
	check("VkMemoryAllocateInfo", unsafe.Sizeof(ma), 32,
		field{"memoryTypeIndex", o(unsafe.Pointer(&ma), unsafe.Pointer(&ma.typeIdx)), 24})
	check("VkDescriptorSetLayoutBinding", unsafe.Sizeof(db), 24,
		field{"pImmutableSamplers", o(unsafe.Pointer(&db), unsafe.Pointer(&db.pSamplers)), 16})
	check("VkDescriptorSetLayoutCreateInfo", unsafe.Sizeof(dl), 32,
		field{"pBindings", o(unsafe.Pointer(&dl), unsafe.Pointer(&dl.pBind)), 24})
	check("VkPipelineLayoutCreateInfo", unsafe.Sizeof(pl), 48,
		field{"pSetLayouts", o(unsafe.Pointer(&pl), unsafe.Pointer(&pl.pSets)), 24},
		field{"pPushConstantRanges", o(unsafe.Pointer(&pl), unsafe.Pointer(&pl.pPush)), 40})
	check("VkShaderModuleCreateInfo", unsafe.Sizeof(sm), 40,
		field{"codeSize", o(unsafe.Pointer(&sm), unsafe.Pointer(&sm.size)), 24},
		field{"pCode", o(unsafe.Pointer(&sm), unsafe.Pointer(&sm.pCode)), 32})
	check("VkPipelineShaderStageCreateInfo", unsafe.Sizeof(st), 48,
		field{"module", o(unsafe.Pointer(&st), unsafe.Pointer(&st.module)), 24},
		field{"pName", o(unsafe.Pointer(&st), unsafe.Pointer(&st.pName)), 32})
	check("VkComputePipelineCreateInfo", unsafe.Sizeof(cp), 96,
		field{"stage", o(unsafe.Pointer(&cp), unsafe.Pointer(&cp.stage)), 24},
		field{"layout", o(unsafe.Pointer(&cp), unsafe.Pointer(&cp.layout)), 72},
		field{"basePipelineIndex", o(unsafe.Pointer(&cp), unsafe.Pointer(&cp.baseIdx)), 88})
	check("VkDescriptorPoolCreateInfo", unsafe.Sizeof(dp), 40,
		field{"pPoolSizes", o(unsafe.Pointer(&dp), unsafe.Pointer(&dp.pSizes)), 32})
	check("VkDescriptorSetAllocateInfo", unsafe.Sizeof(da), 40,
		field{"pSetLayouts", o(unsafe.Pointer(&da), unsafe.Pointer(&da.pSets)), 32})
	check("VkWriteDescriptorSet", unsafe.Sizeof(wd), 64,
		field{"dstSet", o(unsafe.Pointer(&wd), unsafe.Pointer(&wd.dstSet)), 16},
		field{"descriptorType", o(unsafe.Pointer(&wd), unsafe.Pointer(&wd.kind)), 36},
		field{"pBufferInfo", o(unsafe.Pointer(&wd), unsafe.Pointer(&wd.pBuffer)), 48})
	check("VkCommandPoolCreateInfo", unsafe.Sizeof(cpc), 24,
		field{"queueFamilyIndex", o(unsafe.Pointer(&cpc), unsafe.Pointer(&cpc.family)), 20})
	check("VkCommandBufferAllocateInfo", unsafe.Sizeof(cba), 32,
		field{"commandBufferCount", o(unsafe.Pointer(&cba), unsafe.Pointer(&cba.count)), 28})
	check("VkCommandBufferBeginInfo", unsafe.Sizeof(cbb), 32,
		field{"pInheritanceInfo", o(unsafe.Pointer(&cbb), unsafe.Pointer(&cbb.pInherit)), 24})
	check("VkSubmitInfo", unsafe.Sizeof(si), 72,
		field{"pCommandBuffers", o(unsafe.Pointer(&si), unsafe.Pointer(&si.pCmd)), 48},
		field{"pSignalSemaphores", o(unsafe.Pointer(&si), unsafe.Pointer(&si.pSignal)), 64})
	// The subgroup-width structs: every one is a number the engine selects a
	// kernel on. minSubgroupSize read at maxSubgroupSize's offset would turn an
	// 8/32 device into a false 32/32 guarantee.
	check("VkPhysicalDeviceProperties2", unsafe.Sizeof(pp2), 840,
		field{"pNext", o(unsafe.Pointer(&pp2), unsafe.Pointer(&pp2.pNext)), 8},
		field{"properties", o(unsafe.Pointer(&pp2), unsafe.Pointer(&pp2.props)), 16})
	check("VkPhysicalDeviceSubgroupProperties", unsafe.Sizeof(sgp), 32,
		field{"pNext", o(unsafe.Pointer(&sgp), unsafe.Pointer(&sgp.pNext)), 8},
		field{"subgroupSize", o(unsafe.Pointer(&sgp), unsafe.Pointer(&sgp.size)), 16},
		field{"supportedStages", o(unsafe.Pointer(&sgp), unsafe.Pointer(&sgp.stages)), 20},
		field{"supportedOperations", o(unsafe.Pointer(&sgp), unsafe.Pointer(&sgp.ops)), 24},
		field{"quadOperationsInAllStages", o(unsafe.Pointer(&sgp), unsafe.Pointer(&sgp.quadAll)), 28})
	check("VkPhysicalDeviceSubgroupSizeControlProperties", unsafe.Sizeof(scp), 32,
		field{"pNext", o(unsafe.Pointer(&scp), unsafe.Pointer(&scp.pNext)), 8},
		field{"minSubgroupSize", o(unsafe.Pointer(&scp), unsafe.Pointer(&scp.min)), 16},
		field{"maxSubgroupSize", o(unsafe.Pointer(&scp), unsafe.Pointer(&scp.max)), 20},
		field{"maxComputeWorkgroupSubgroups", o(unsafe.Pointer(&scp), unsafe.Pointer(&scp.maxGroups)), 24},
		field{"requiredSubgroupSizeStages", o(unsafe.Pointer(&scp), unsafe.Pointer(&scp.reqStages)), 28})
	check("VkPhysicalDeviceSubgroupSizeControlFeatures", unsafe.Sizeof(scf), 24,
		field{"pNext", o(unsafe.Pointer(&scf), unsafe.Pointer(&scf.pNext)), 8},
		field{"subgroupSizeControl", o(unsafe.Pointer(&scf), unsafe.Pointer(&scf.control)), 16},
		field{"computeFullSubgroups", o(unsafe.Pointer(&scf), unsafe.Pointer(&scf.full)), 20})
	check("VkPipelineShaderStageRequiredSubgroupSizeCreateInfo", unsafe.Sizeof(rsc), 24,
		field{"pNext", o(unsafe.Pointer(&rsc), unsafe.Pointer(&rsc.pNext)), 8},
		field{"requiredSubgroupSize", o(unsafe.Pointer(&rsc), unsafe.Pointer(&rsc.size)), 16})
	check("VkDescriptorBufferInfo", unsafe.Sizeof(descBufInfo{}), 24)
	check("VkQueueFamilyProperties", unsafe.Sizeof(queueFamily{}), 24)
	check("VkMemoryRequirements", unsafe.Sizeof(memReq{}), 24)
	check("VkDescriptorPoolSize", unsafe.Sizeof(poolSize{}), 8)
	var pr pushRange
	check("VkPushConstantRange", unsafe.Sizeof(pr), 12,
		field{"offset", o(unsafe.Pointer(&pr), unsafe.Pointer(&pr.offset)), 4},
		field{"size", o(unsafe.Pointer(&pr), unsafe.Pointer(&pr.size)), 8})
}

// TestWireConstants pins the enum values device.go and Mem branch on.
//
// A wrong value does not fail, it re-ranks or mis-queries silently. The
// numbers are from vulkan_core.h, not from this file.
func TestWireConstants(t *testing.T) {
	for _, c := range []struct {
		name string
		got  int
		want int
	}{
		{"VK_PHYSICAL_DEVICE_TYPE_OTHER", devTypeOther, 0},
		{"VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU", devTypeIntegrated, 1},
		{"VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU", devTypeDiscrete, 2},
		{"VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU", devTypeVirtual, 3},
		{"VK_PHYSICAL_DEVICE_TYPE_CPU", devTypeCPU, 4},
		{"VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2", stPhysDevMemProps2, 1000059006},
		{"VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT", stMemBudgetEXT, 1000237000},
		{"VK_MEMORY_HEAP_DEVICE_LOCAL_BIT", heapDeviceLocal, 1},
		{"VK_MAX_MEMORY_HEAPS", maxMemoryHeaps, 16},
		{"VK_MAX_PHYSICAL_DEVICE_NAME_SIZE", maxDevNameLen, 256},
		{"VK_MAX_EXTENSION_NAME_SIZE", maxExtNameLen, 256},
		{"VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2", stPhysDevProps2, 1000059001},
		{"VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES", stSubgroupProps, 1000094000},
		{"VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES", stSizeCtlProps, 1000225000},
		{"VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO", stReqSizeCI, 1000225001},
		{"VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES", stSizeCtlFeat, 1000225002},
		{"VK_SUBGROUP_FEATURE_BASIC_BIT", subgroupBasic, 0x00000001},
		{"VK_SUBGROUP_FEATURE_SHUFFLE_BIT", subgroupShuffle, 0x00000010},
		{"VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT", pipeStageRequireFullSubgroups, 0x00000002},
		{"VK_SHADER_STAGE_COMPUTE_BIT", stageCompute, 0x00000020},
		{"VK_API_VERSION_1_1", apiVersion11, 4198400},

		// The cooperative-matrix three. Extension 507's base is 1000506000 and
		// the features struct is the first. A pNext entry with an unknown sType
		// is ignored by specification, so a wrong value reads as a silent zero.
		{"VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR", stCoopMatFeat, 1000506000},
		{"VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR", structTypeCoopMatProps, 1000506001},
		{"VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2", stPhysDevFeat2, 1000059000},
		{"VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES", stStorage16, 1000083000},
		{"VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES", stStorage8, 1000177000},
		{"VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES", stShaderF16I8, 1000082000},
	} {
		if c.got != c.want {
			t.Errorf("%s = %d, C says %d", c.name, c.got, c.want)
		}
	}
	if extCooperativeMatrix != "VK_KHR_cooperative_matrix" {
		t.Errorf("extCooperativeMatrix = %q", extCooperativeMatrix)
	}
	if extMemoryBudget != "VK_EXT_memory_budget" {
		t.Errorf("extension name %q, C says VK_EXT_memory_budget", extMemoryBudget)
	}
	if extSubgroupSizeControl != "VK_EXT_subgroup_size_control" {
		t.Errorf("extension name %q, C says VK_EXT_subgroup_size_control", extSubgroupSizeControl)
	}
	// The ranking, stated as an order rather than as five separate numbers.
	if !(devTypeRank(devTypeDiscrete) > devTypeRank(devTypeIntegrated) &&
		devTypeRank(devTypeIntegrated) > devTypeRank(devTypeVirtual) &&
		devTypeRank(devTypeVirtual) > devTypeRank(devTypeOther) &&
		devTypeRank(devTypeOther) > 0 && devTypeRank(devTypeCPU) < 0) {
		t.Errorf("rank order wrong: discrete %d integrated %d virtual %d other %d cpu %d",
			devTypeRank(devTypeDiscrete), devTypeRank(devTypeIntegrated),
			devTypeRank(devTypeVirtual), devTypeRank(devTypeOther), devTypeRank(devTypeCPU))
	}
}
