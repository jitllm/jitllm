package hip

import (
	"fmt"
	"strconv"
	"strings"
	"unsafe"

	"github.com/jitllm/jitllm/backend/ffi"
)

// comgr's handles are structs of one uint64; on the C ABIs jitllm builds for
// such a struct is passed and returned exactly as the integer is.
type (
	cgStatus int32
	cgData   uint64
	cgSet    uint64
	cgInfo   uint64
	cgMeta   uint64
)

// The amd_comgr.h constants this package uses.
const (
	cgDataSource     = 0x1
	cgDataLog        = 0x5
	cgDataBC         = 0x6
	cgDataRelocable  = 0x7
	cgDataExecutable = 0x8

	cgMetaString = 0x1
	cgMetaMap    = 0x2
	cgMetaList   = 0x3
)

// The action kinds this package names. They are not the library's numbers,
// which moved between API versions: see comgrCodes.
const (
	cgActCompileSourceToBC = iota
	cgActCodegenBCToReloc
	cgActCodegenBCToAsm
	cgActLinkRelocToExec
	cgActAssembleSourceToRel
)

// comgrCodes is one API version's numbering of the actions and of
// AMD_COMGR_LANGUAGE_LLVM_IR, read off each version's amd_comgr.h. Comgr 3
// (ROCm 6.4 and later) removed ADD_DEVICE_LIBRARIES and OPTIMIZE_BC_TO_BC and
// renumbered everything after them, and dropped a language; comgr 2 is what
// ROCm 6.0 to 6.3 ship on Linux and what AMD's Windows driver ships
// (amd_comgr_2.dll). The data and metadata kinds are the same in both.
type comgrCodes struct {
	act    [5]int32 // indexed by the cgAct* kinds
	llvmIR int32
}

var (
	comgrV2 = comgrCodes{act: [5]int32{0x2, 0x6, 0x7, 0x9, 0xA}, llvmIR: 0x5}
	comgrV3 = comgrCodes{act: [5]int32{0x2, 0x4, 0x5, 0x7, 0x8}, llvmIR: 0x4}
)

type comgrAPI struct {
	major, minor int
	codes        comgrCodes

	statusString  func(cgStatus, *unsafe.Pointer) cgStatus
	createData    func(int32, *cgData) cgStatus
	releaseData   func(cgData) cgStatus
	setData       func(cgData, uint64, *byte) cgStatus
	setDataName   func(cgData, *byte) cgStatus
	getData       func(cgData, *uint64, *byte) cgStatus
	createSet     func(*cgSet) cgStatus
	destroySet    func(cgSet) cgStatus
	setAdd        func(cgSet, cgData) cgStatus
	dataCount     func(cgSet, int32, *uint64) cgStatus
	dataGet       func(cgSet, int32, uint64, *cgData) cgStatus
	createInfo    func(*cgInfo) cgStatus
	destroyInfo   func(cgInfo) cgStatus
	setISA        func(cgInfo, *byte) cgStatus
	setLanguage   func(cgInfo, int32) cgStatus
	setOptions    func(cgInfo, *unsafe.Pointer, uint64) cgStatus
	setLogging    func(cgInfo, uint8) cgStatus
	doAction      func(int32, cgInfo, cgSet, cgSet) cgStatus
	getMeta       func(cgData, *cgMeta) cgStatus
	destroyMeta   func(cgMeta) cgStatus
	metaKind      func(cgMeta, *int32) cgStatus
	metaString    func(cgMeta, *uint64, *byte) cgStatus
	metaLookup    func(cgMeta, *byte, *cgMeta) cgStatus
	metaListSize  func(cgMeta, *uint64) cgStatus
	metaListIndex func(cgMeta, uint64, *cgMeta) cgStatus
	isaCount      func(*uint64) cgStatus
	isaName       func(uint64, *unsafe.Pointer) cgStatus
}

func bindComgr(names []string) (*comgrAPI, error) {
	l, err := openFirst("libamd_comgr", names)
	if err != nil {
		return nil, err
	}
	a := &comgrAPI{}
	err = bindSafely("libamd_comgr", func() {
		getVersion := ffi.Fn2[ffi.None, *uint64, *uint64](l, "amd_comgr_get_version")
		var maj, min uint64
		getVersion(&maj, &min)
		a.major, a.minor = int(maj), int(min)
		a.statusString = ffi.Fn2[cgStatus, cgStatus, *unsafe.Pointer](l, "amd_comgr_status_string")
		a.createData = ffi.Fn2[cgStatus, int32, *cgData](l, "amd_comgr_create_data")
		a.releaseData = ffi.Fn1[cgStatus, cgData](l, "amd_comgr_release_data")
		a.setData = ffi.Fn3[cgStatus, cgData, uint64, *byte](l, "amd_comgr_set_data")
		a.setDataName = ffi.Fn2[cgStatus, cgData, *byte](l, "amd_comgr_set_data_name")
		a.getData = ffi.Fn3[cgStatus, cgData, *uint64, *byte](l, "amd_comgr_get_data")
		a.createSet = ffi.Fn1[cgStatus, *cgSet](l, "amd_comgr_create_data_set")
		a.destroySet = ffi.Fn1[cgStatus, cgSet](l, "amd_comgr_destroy_data_set")
		a.setAdd = ffi.Fn2[cgStatus, cgSet, cgData](l, "amd_comgr_data_set_add")
		a.dataCount = ffi.Fn3[cgStatus, cgSet, int32, *uint64](l, "amd_comgr_action_data_count")
		a.dataGet = ffi.Fn4[cgStatus, cgSet, int32, uint64, *cgData](l, "amd_comgr_action_data_get_data")
		a.createInfo = ffi.Fn1[cgStatus, *cgInfo](l, "amd_comgr_create_action_info")
		a.destroyInfo = ffi.Fn1[cgStatus, cgInfo](l, "amd_comgr_destroy_action_info")
		a.setISA = ffi.Fn2[cgStatus, cgInfo, *byte](l, "amd_comgr_action_info_set_isa_name")
		a.setLanguage = ffi.Fn2[cgStatus, cgInfo, int32](l, "amd_comgr_action_info_set_language")
		a.setOptions = ffi.Fn3[cgStatus, cgInfo, *unsafe.Pointer, uint64](l, "amd_comgr_action_info_set_option_list")
		a.setLogging = ffi.Fn2[cgStatus, cgInfo, uint8](l, "amd_comgr_action_info_set_logging")
		a.doAction = ffi.Fn4[cgStatus, int32, cgInfo, cgSet, cgSet](l, "amd_comgr_do_action")
		a.getMeta = ffi.Fn2[cgStatus, cgData, *cgMeta](l, "amd_comgr_get_data_metadata")
		a.destroyMeta = ffi.Fn1[cgStatus, cgMeta](l, "amd_comgr_destroy_metadata")
		a.metaKind = ffi.Fn2[cgStatus, cgMeta, *int32](l, "amd_comgr_get_metadata_kind")
		a.metaString = ffi.Fn3[cgStatus, cgMeta, *uint64, *byte](l, "amd_comgr_get_metadata_string")
		a.metaLookup = ffi.Fn3[cgStatus, cgMeta, *byte, *cgMeta](l, "amd_comgr_metadata_lookup")
		a.metaListSize = ffi.Fn2[cgStatus, cgMeta, *uint64](l, "amd_comgr_get_metadata_list_size")
		a.metaListIndex = ffi.Fn3[cgStatus, cgMeta, uint64, *cgMeta](l, "amd_comgr_index_list_metadata")
		a.isaCount = ffi.Fn1[cgStatus, *uint64](l, "amd_comgr_get_isa_count")
		a.isaName = ffi.Fn2[cgStatus, uint64, *unsafe.Pointer](l, "amd_comgr_get_isa_name")
	})
	if err != nil {
		return nil, err
	}
	switch {
	case a.major < 2:
		return nil, fmt.Errorf("hip: libamd_comgr %d.%d is older than the 2.x API this binding uses", a.major, a.minor)
	case a.major == 2:
		a.codes = comgrV2
	case a.major == 3:
		a.codes = comgrV3
	default:
		// A later API may renumber again, as 3 did: refused rather than
		// driven with numbers that may now mean other actions.
		return nil, fmt.Errorf("hip: libamd_comgr %d.%d is newer than the 2.x and 3.x APIs this binding knows", a.major, a.minor)
	}
	return a, nil
}

// cstr is s NUL-terminated, as a byte slice the caller keeps alive across the
// call that reads it.
func cstr(s string) []byte { return append([]byte(s), 0) }

// goString copies a NUL-terminated C string the library owns.
func goString(p unsafe.Pointer) string {
	if p == nil {
		return ""
	}
	var b []byte
	for i := 0; ; i++ {
		c := *(*byte)(unsafe.Add(p, i))
		if c == 0 {
			return string(b)
		}
		b = append(b, c)
	}
}

func (a *comgrAPI) err(s cgStatus, what string) error {
	if s == 0 {
		return nil
	}
	var p unsafe.Pointer
	msg := ""
	if a.statusString(s, &p) == 0 {
		msg = goString(p)
	}
	return fmt.Errorf("hip: comgr: %s: %s (status %d)", what, msg, s)
}

// bytesOf reads a data object's contents.
func (a *comgrAPI) bytesOf(d cgData) ([]byte, error) {
	var n uint64
	if err := a.err(a.getData(d, &n, nil), "get_data size"); err != nil {
		return nil, err
	}
	if n == 0 {
		return nil, nil
	}
	out := make([]byte, n)
	if err := a.err(a.getData(d, &n, &out[0]), "get_data"); err != nil {
		return nil, err
	}
	return out[:n], nil
}

// ISAs is every target this comgr can generate code for, in its own spelling
// (amdgcn-amd-amdhsa--gfx90a and so on).
func (a *comgrAPI) isas() []string {
	var n uint64
	if a.isaCount(&n) != 0 {
		return nil
	}
	out := make([]string, 0, n)
	for i := uint64(0); i < n; i++ {
		var p unsafe.Pointer
		if a.isaName(i, &p) == 0 {
			out = append(out, goString(p))
		}
	}
	return out
}

// Comgr is a loaded code object manager.
type Comgr struct{ a *comgrAPI }

// LoadComgr loads comgr alone. It is all an offline gate needs: comgr generates
// code for any target with no GPU present.
func LoadComgr(c Config) (*Comgr, error) {
	l := load(c)
	if l.comgrErr != nil {
		return nil, l.comgrErr
	}
	return &Comgr{a: l.comgr}, nil
}

// Version is comgr's API version.
func (c *Comgr) Version() (major, minor int) { return c.a.major, c.a.minor }

// ISAs lists every target this comgr supports.
func (c *Comgr) ISAs() []string { return c.a.isas() }

// ISAName is comgr's name for a target as HIP's gcnArchName spells it
// ("gfx90a:sramecc+:xnack-").
func ISAName(arch string) string { return "amdgcn-amd-amdhsa--" + arch }

// CodeObject is an executable a HIP module loads, with what its metadata says
// about each kernel in it.
type CodeObject struct {
	Bytes   []byte
	Kernels []KernelMeta
}

// KernelMeta is one kernel's resource usage, as the code object's own
// metadata (the amdhsa.kernels note) states it. Nothing here is estimated.
type KernelMeta struct {
	Name string
	// SGPRs, VGPRs and AGPRs are the registers one wave uses.
	SGPRs, VGPRs, AGPRs int
	// Scratch is the private segment per work-item in bytes: anything above
	// zero is a spill or a stack object, which jitllm's kernels never ask for.
	Scratch int
	// LDS is the workgroup's shared memory in bytes.
	LDS int
	// Wave is the wavefront size the kernel was generated for: 64 on CDNA, 32
	// on RDNA unless asked otherwise.
	Wave int
	// SGPRSpills and VGPRSpills count the registers spilled.
	SGPRSpills, VGPRSpills int
	// MaxGroup is the largest workgroup the kernel may be launched with.
	MaxGroup int
}

// Compile turns LLVM IR text into a loadable code object for one target, all
// inside comgr: the IR is parsed and optimised (COMPILE_SOURCE_TO_BC), given
// to the AMDGPU code generator (CODEGEN_BC_TO_RELOCATABLE) and linked
// (LINK_RELOCATABLE_TO_EXECUTABLE). isa is HIP's gcnArchName, features
// included.
func (c *Comgr) Compile(llvmIR, isa string) (*CodeObject, error) {
	bc, err := c.action(cgActCompileSourceToBC, isa, []string{"-O3"}, cgDataSource, "k.ll", []byte(llvmIR), cgDataBC)
	if err != nil {
		return nil, err
	}
	rel, err := c.action(cgActCodegenBCToReloc, isa, []string{"-O3"}, cgDataBC, "k.bc", bc, cgDataRelocable)
	if err != nil {
		return nil, err
	}
	exe, err := c.action(cgActLinkRelocToExec, isa, nil, cgDataRelocable, "k.o", rel, cgDataExecutable)
	if err != nil {
		return nil, err
	}
	co := &CodeObject{Bytes: exe}
	co.Kernels, err = c.kernelMeta(exe)
	if err != nil {
		return nil, err
	}
	return co, nil
}

// Assembly is the AMDGPU assembly comgr generates for LLVM IR text, for a
// reader: what the lowering asked for, as instructions.
func (c *Comgr) Assembly(llvmIR, isa string) (string, error) {
	bc, err := c.action(cgActCompileSourceToBC, isa, []string{"-O3"}, cgDataSource, "k.ll", []byte(llvmIR), cgDataBC)
	if err != nil {
		return "", err
	}
	s, err := c.action(cgActCodegenBCToAsm, isa, []string{"-O3"}, cgDataBC, "k.bc", bc, cgDataSource)
	return string(s), err
}

// action runs one comgr action over one input and returns its one output of
// kind out. A failure carries comgr's log, which is the compiler's own
// diagnostic.
func (c *Comgr) action(kind int32, isa string, opts []string, inKind int32, inName string, in []byte, out int32) ([]byte, error) {
	a := c.a
	var data cgData
	if err := a.err(a.createData(inKind, &data), "create_data"); err != nil {
		return nil, err
	}
	defer a.releaseData(data)
	if len(in) == 0 {
		return nil, fmt.Errorf("hip: comgr: empty input")
	}
	if err := a.err(a.setData(data, uint64(len(in)), &in[0]), "set_data"); err != nil {
		return nil, err
	}
	nm := cstr(inName)
	if err := a.err(a.setDataName(data, &nm[0]), "set_data_name"); err != nil {
		return nil, err
	}
	var inSet, outSet cgSet
	if err := a.err(a.createSet(&inSet), "create_data_set"); err != nil {
		return nil, err
	}
	defer a.destroySet(inSet)
	if err := a.err(a.createSet(&outSet), "create_data_set"); err != nil {
		return nil, err
	}
	defer a.destroySet(outSet)
	if err := a.err(a.setAdd(inSet, data), "data_set_add"); err != nil {
		return nil, err
	}
	var info cgInfo
	if err := a.err(a.createInfo(&info), "create_action_info"); err != nil {
		return nil, err
	}
	defer a.destroyInfo(info)
	name := cstr(ISAName(isa))
	if err := a.err(a.setISA(info, &name[0]), "set_isa_name "+isa); err != nil {
		return nil, err
	}
	if kind == cgActCompileSourceToBC {
		if err := a.err(a.setLanguage(info, a.codes.llvmIR), "set_language"); err != nil {
			return nil, err
		}
	}
	if len(opts) > 0 {
		// The option strings and the array of pointers to them stay reachable
		// for the call: cs holds the bytes, ps the addresses.
		cs := make([][]byte, len(opts))
		ps := make([]unsafe.Pointer, len(opts))
		for i, o := range opts {
			cs[i] = cstr(o)
			ps[i] = unsafe.Pointer(&cs[i][0])
		}
		err := a.err(a.setOptions(info, &ps[0], uint64(len(ps))), "set_option_list")
		if err != nil {
			return nil, err
		}
	}
	if err := a.err(a.setLogging(info, 1), "set_logging"); err != nil {
		return nil, err
	}
	st := a.doAction(a.codes.act[kind], info, inSet, outSet)
	log := c.log(outSet)
	if st != 0 {
		return nil, fmt.Errorf("%w\n%s", a.err(st, actionName(kind)+" for "+isa), log)
	}
	var n uint64
	if err := a.err(a.dataCount(outSet, out, &n), "action_data_count"); err != nil {
		return nil, err
	}
	if n != 1 {
		return nil, fmt.Errorf("hip: comgr: %s for %s produced %d outputs, want 1\n%s", actionName(kind), isa, n, log)
	}
	var od cgData
	if err := a.err(a.dataGet(outSet, out, 0, &od), "action_data_get_data"); err != nil {
		return nil, err
	}
	defer a.releaseData(od)
	return a.bytesOf(od)
}

// log is the text of every log object an action left in its result set.
func (c *Comgr) log(set cgSet) string {
	a := c.a
	var n uint64
	if a.dataCount(set, cgDataLog, &n) != 0 {
		return ""
	}
	var sb strings.Builder
	for i := uint64(0); i < n; i++ {
		var d cgData
		if a.dataGet(set, cgDataLog, i, &d) != 0 {
			continue
		}
		b, err := a.bytesOf(d)
		a.releaseData(d)
		if err == nil {
			sb.Write(b)
		}
	}
	return strings.TrimSpace(sb.String())
}

func actionName(kind int32) string {
	switch kind {
	case cgActCompileSourceToBC:
		return "COMPILE_SOURCE_TO_BC"
	case cgActCodegenBCToReloc:
		return "CODEGEN_BC_TO_RELOCATABLE"
	case cgActCodegenBCToAsm:
		return "CODEGEN_BC_TO_ASSEMBLY"
	case cgActLinkRelocToExec:
		return "LINK_RELOCATABLE_TO_EXECUTABLE"
	case cgActAssembleSourceToRel:
		return "ASSEMBLE_SOURCE_TO_RELOCATABLE"
	}
	return "action " + strconv.Itoa(int(kind))
}

// kernelMeta reads the amdhsa.kernels list out of an executable's metadata.
func (c *Comgr) kernelMeta(exe []byte) ([]KernelMeta, error) {
	a := c.a
	var d cgData
	if err := a.err(a.createData(cgDataExecutable, &d), "create_data"); err != nil {
		return nil, err
	}
	defer a.releaseData(d)
	if err := a.err(a.setData(d, uint64(len(exe)), &exe[0]), "set_data"); err != nil {
		return nil, err
	}
	var root cgMeta
	if err := a.err(a.getMeta(d, &root), "get_data_metadata"); err != nil {
		return nil, err
	}
	defer a.destroyMeta(root)
	ks, ok := c.lookup(root, "amdhsa.kernels")
	if !ok {
		return nil, fmt.Errorf("hip: comgr: the code object's metadata has no amdhsa.kernels")
	}
	defer a.destroyMeta(ks)
	var n uint64
	if err := a.err(a.metaListSize(ks, &n), "get_metadata_list_size"); err != nil {
		return nil, err
	}
	var out []KernelMeta
	for i := uint64(0); i < n; i++ {
		var k cgMeta
		if err := a.err(a.metaListIndex(ks, i, &k), "index_list_metadata"); err != nil {
			return nil, err
		}
		m := KernelMeta{
			Name:       c.str(k, ".name"),
			SGPRs:      c.num(k, ".sgpr_count"),
			VGPRs:      c.num(k, ".vgpr_count"),
			AGPRs:      c.num(k, ".agpr_count"),
			Scratch:    c.num(k, ".private_segment_fixed_size"),
			LDS:        c.num(k, ".group_segment_fixed_size"),
			Wave:       c.num(k, ".wavefront_size"),
			SGPRSpills: c.num(k, ".sgpr_spill_count"),
			VGPRSpills: c.num(k, ".vgpr_spill_count"),
			MaxGroup:   c.num(k, ".max_flat_workgroup_size"),
		}
		a.destroyMeta(k)
		out = append(out, m)
	}
	return out, nil
}

func (c *Comgr) lookup(m cgMeta, key string) (cgMeta, bool) {
	k := cstr(key)
	var v cgMeta
	if c.a.metaLookup(m, &k[0], &v) != 0 {
		return 0, false
	}
	return v, true
}

// str is a string value of a map, or "" when the key is absent.
func (c *Comgr) str(m cgMeta, key string) string {
	v, ok := c.lookup(m, key)
	if !ok {
		return ""
	}
	defer c.a.destroyMeta(v)
	var kind int32
	if c.a.metaKind(v, &kind) != 0 || kind != cgMetaString {
		return ""
	}
	var n uint64
	if c.a.metaString(v, &n, nil) != 0 || n == 0 {
		return ""
	}
	b := make([]byte, n)
	if c.a.metaString(v, &n, &b[0]) != 0 {
		return ""
	}
	return strings.TrimRight(string(b), "\x00")
}

// num is an integer value of a map, or -1 when the key is absent: absent and
// zero are different facts (an agpr count on a target with no AGPRs).
func (c *Comgr) num(m cgMeta, key string) int {
	s := c.str(m, key)
	if s == "" {
		return -1
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return -1
	}
	return n
}
