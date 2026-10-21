package vulkan

import (
	"fmt"
	"unsafe"

	"github.com/jitllm/jitllm/jit_old/gpu/ffi"
)

// CoopMatShape is one matrix shape and type combination this device supports,
// asked rather than assumed. `cooperativeMatrix = true` is only a feature bit;
// the usable combinations are a list the driver enumerates. ir.OpMMA's
// PTX-shaped tiles (NVIDIA's mma.sync 16x8x16, 16x8x32) are not that list.
type CoopMatShape struct {
	M, N, K                int
	AType, BType           ComponentType
	CType, ResultType      ComponentType
	SaturatingAccumulation bool
	Scope                  Scope
}

// ComponentType is VkComponentTypeKHR.
type ComponentType uint32

const (
	CompF16 ComponentType = 0
	CompF32 ComponentType = 1
	CompF64 ComponentType = 2
	CompS8  ComponentType = 3
	CompS16 ComponentType = 4
	CompS32 ComponentType = 5
	CompS64 ComponentType = 6
	CompU8  ComponentType = 7
	CompU16 ComponentType = 8
	CompU32 ComponentType = 9
	CompU64 ComponentType = 10
)

func (c ComponentType) String() string {
	switch c {
	case CompF16:
		return "f16"
	case CompF32:
		return "f32"
	case CompF64:
		return "f64"
	case CompS8:
		return "s8"
	case CompS16:
		return "s16"
	case CompS32:
		return "s32"
	case CompS64:
		return "s64"
	case CompU8:
		return "u8"
	case CompU16:
		return "u16"
	case CompU32:
		return "u32"
	case CompU64:
		return "u64"
	}
	return fmt.Sprintf("component(%d)", uint32(c))
}

// Scope is VkScopeKHR. Subgroup is the only one a warp-level op can use.
type Scope uint32

const (
	ScopeDevice      Scope = 1
	ScopeWorkgroup   Scope = 2
	ScopeSubgroup    Scope = 3
	ScopeQueueFamily Scope = 5
)

func (s Scope) String() string {
	switch s {
	case ScopeDevice:
		return "device"
	case ScopeWorkgroup:
		return "workgroup"
	case ScopeSubgroup:
		return "subgroup"
	case ScopeQueueFamily:
		return "queuefamily"
	}
	return fmt.Sprintf("scope(%d)", uint32(s))
}

func (s CoopMatShape) String() string {
	sat := ""
	if s.SaturatingAccumulation {
		sat = " saturating"
	}
	return fmt.Sprintf("%dx%dx%d %s*%s+%s->%s %s%s",
		s.M, s.N, s.K, s.AType, s.BType, s.CType, s.ResultType, s.Scope, sat)
}

// coopMatPropsBytes is sizeof(VkCooperativeMatrixPropertiesKHR): sType and four
// bytes of padding, pNext, three u32 sizes, four component types, a VkBool32 and
// a scope. The driver writes sType itself only if the caller sets it, which is
// why every element is initialised below rather than zeroed once.
const coopMatPropsBytes = 56

// structTypeCoopMatProps is VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR.
const structTypeCoopMatProps = 1000506001

// CooperativeMatrixShapes enumerates the (M,N,K,types,scope) combinations this
// physical device supports, or nil when the extension is absent.
//
// It goes through vkGetInstanceProcAddr, because the loader does not export
// this symbol (the instance-side counterpart of ffi.Fn4At's device case).
func CooperativeMatrixShapes(inst Instance, phys PhysDevice) ([]CoopMatShape, error) {
	if err := load(); err != nil {
		return nil, err
	}
	name := append([]byte("vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"), 0)
	fn := vkGetInstanceProcAddr(inst, unsafe.Pointer(&name[0]))
	if fn == nil {
		// Not an error: a device without the extension is the ordinary case and
		// the caller decides what to do about it.
		return nil, nil
	}
	get := ffi.Fn3At[Result, PhysDevice, *uint32, up](fn,
		"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR")

	var n uint32
	if r := get(phys, &n, nil); r != 0 {
		return nil, fmt.Errorf("vulkan: cooperative matrix count: %v", r)
	}
	if n == 0 {
		return nil, nil
	}
	buf := make([]byte, int(n)*coopMatPropsBytes)
	for i := 0; i < int(n); i++ {
		// sType must be set on every element before the call: the driver reads
		// it to validate the chain and a zeroed one is not this structure.
		put32(buf[i*coopMatPropsBytes:], structTypeCoopMatProps)
	}
	if r := get(phys, &n, up(unsafe.Pointer(&buf[0]))); r != 0 {
		return nil, fmt.Errorf("vulkan: cooperative matrix properties: %v", r)
	}
	out := make([]CoopMatShape, 0, n)
	for i := 0; i < int(n); i++ {
		b := buf[i*coopMatPropsBytes:]
		out = append(out, CoopMatShape{
			M: int(get32(b, 16)), N: int(get32(b, 20)), K: int(get32(b, 24)),
			AType: ComponentType(get32(b, 28)), BType: ComponentType(get32(b, 32)),
			CType: ComponentType(get32(b, 36)), ResultType: ComponentType(get32(b, 40)),
			SaturatingAccumulation: get32(b, 44) != 0,
			Scope:                  Scope(get32(b, 48)),
		})
	}
	return out, nil
}

func put32(b []byte, v uint32) {
	b[0], b[1], b[2], b[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
}

func get32(b []byte, off int) uint32 {
	return uint32(b[off]) | uint32(b[off+1])<<8 | uint32(b[off+2])<<16 | uint32(b[off+3])<<24
}

// tileFeats are the device features behind the five SPIR-V capabilities the
// tile kernels declare: CooperativeMatrixKHR, Float16,
// StorageBuffer16BitAccess, Int8 and StorageBuffer8BitAccess. Each needs its
// VkBool32 set at vkCreateDevice. A permissive driver may run a module without
// them and get one format right and another silently wrong, so they are
// queried (vkGetPhysicalDeviceFeatures2) and enabled, not assumed.
type tileFeats struct {
	coop   coopMatFeat
	s16    storage16Feat
	s8     storage8Feat
	f16i8  shaderF16I8Feat
	CoopOK bool // every feature the tile kernels need came back supported
}

type coopMatFeat struct {
	sType  uint32
	_      uint32
	pNext  uintptr
	on     uint32 // cooperativeMatrix
	robust uint32 // cooperativeMatrixRobustBufferAccess
}

type storage16Feat struct {
	sType    uint32
	_        uint32
	pNext    uintptr
	buffer   uint32 // storageBuffer16BitAccess
	uniform  uint32
	pushCnst uint32
	inputOut uint32
}

type storage8Feat struct {
	sType    uint32
	_        uint32
	pNext    uintptr
	buffer   uint32 // storageBuffer8BitAccess
	uniform  uint32
	pushCnst uint32
}

type shaderF16I8Feat struct {
	sType uint32
	_     uint32
	pNext uintptr
	f16   uint32 // shaderFloat16
	i8    uint32 // shaderInt8
}

// physFeat2 is VkPhysicalDeviceFeatures2. The 55 VkBool32 of the core
// VkPhysicalDeviceFeatures are a slab rather than fields: nothing here reads
// one, and naming fifty-five booleans in a fixed order from memory is
// error-prone.
type physFeat2 struct {
	sType    uint32
	_        uint32
	pNext    uintptr
	features [55]uint32
}

const (
	stPhysDevFeat2 = 1000059000 // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2
	stCoopMatFeat  = 1000506000 // ..._PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
	stStorage16    = 1000083000 // ..._PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES
	stStorage8     = 1000177000 // ..._PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES
	stShaderF16I8  = 1000082000 // ..._PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES

	extCooperativeMatrix = "VK_KHR_cooperative_matrix"
)

// readTileFeats asks the physical device which of the five it supports and
// returns them ready to chain into VkDeviceCreateInfo. It never asks for one
// that came back false: enabling an unsupported feature is a vkCreateDevice
// failure, so a device that cannot do tiles must still open for everything else.
//
// It returns a pointer because the structs link to each other: a pNext chain
// built over a value's fields and returned by value would point into the
// callee's dead frame.
func readTileFeats(p PhysDevice, apiVer uint32) *tileFeats {
	t := new(tileFeats)
	if apiVer < apiVersion11 || vkGetPhysicalDeviceFeatures2 == nil {
		return t
	}
	t.coop.sType, t.s16.sType, t.s8.sType, t.f16i8.sType = stCoopMatFeat, stStorage16, stStorage8, stShaderF16I8
	t.coop.pNext = uintptr(up(&t.s16))
	t.s16.pNext = uintptr(up(&t.s8))
	t.s8.pNext = uintptr(up(&t.f16i8))
	f2 := physFeat2{sType: stPhysDevFeat2, pNext: uintptr(up(&t.coop))}
	vkGetPhysicalDeviceFeatures2(p, up(&f2))

	// Keep only what the tile kernels declare. The driver reports several more
	// in the same structs (robust access, push-constant storage, input/output
	// 16-bit); asking for one the engine never emits widens the contract for
	// nothing and can itself be a refusal on a stricter driver.
	t.coop.robust = 0
	t.s16.uniform, t.s16.pushCnst, t.s16.inputOut = 0, 0, 0
	t.s8.uniform, t.s8.pushCnst = 0, 0
	t.CoopOK = t.coop.on == 1 && t.s16.buffer == 1 && t.s8.buffer == 1 &&
		t.f16i8.f16 == 1 && t.f16i8.i8 == 1
	return t
}

// Tiles reports whether cooperative-matrix kernels may be compiled on this
// device: the extension is present and every feature their SPIR-V declares was
// enabled. A false here is a decline and not a slow path -- the lowering has no
// scalar twin, so a kernel that reaches a device without these computes the
// wrong product rather than a slower one.
func (c *Ctx) Tiles() bool { return c.tiles }

// TileShapes is what this device enumerated at open, cached because the query
// allocates and a Compile may ask per kernel.
func (c *Ctx) TileShapes() []CoopMatShape { return c.shapes }

// SupportsTile reports whether this device enumerated a subgroup-scope
// combination matching an A*B+C over the given tile types.
//
// The driver does not reliably refuse a shape it did not enumerate: it may
// compile the module and return a wrong product, or fail at pipeline creation.
// So the enumeration, not the error, is the check.
func (c *Ctx) SupportsTile(m, n, k int, a, b, acc, result ComponentType) bool {
	for _, s := range c.shapes {
		if s.Scope == ScopeSubgroup && s.M == m && s.N == n && s.K == k &&
			s.AType == a && s.BType == b && s.CType == acc && s.ResultType == result {
			return true
		}
	}
	return false
}
