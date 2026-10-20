package backend_test

import (
	"encoding/binary"
	"fmt"
	"strconv"
	"testing"

	"github.com/jitllm/jitllm/internal/testdevices"
	"github.com/jitllm/jitllm/jit/gpu/backend"
	"github.com/jitllm/jitllm/jit/gpu/ir"
	"github.com/jitllm/jitllm/jit/gpu/kernels"
	"github.com/jitllm/jitllm/jit/gpu/vulkan"
	"github.com/jitllm/jitllm/paritytest/mainkernels"
)

// forcedGroups is the workgroup limit the split gates impose on a Vulkan device
// whose own limit no launch here reaches (2^31-1 on an NVIDIA card). It divides
// none of the launches below, so the last piece is ragged.
const forcedGroups = 97

// specGroups is the Vulkan specification's floor on maxComputeWorkGroupCount
// (65535, an integrated GPU's own limit). A launch past it is the size that splits on
// a device with the smallest limit the specification allows.
const specGroups = 65535

// groupDevice is a device under a split gate, and the limit it was opened with:
// 0 for its own.
type groupDevice struct {
	name   string
	d      backend.Device
	forced uint32
}

// groupDevices is every device this host opens at its own limit -- CUDA,
// Metal, and EVERY Vulkan device by index, not only the one Open's default rule
// picks, since an integrated GPU beside a discrete card is the device with the
// small limit -- and every Vulkan device again with forced lowered to force a
// split where its own limit never would. Software rasterisers are left out:
// llvmpipe shares that limit and takes minutes over these sizes.
func groupDevices(t *testing.T, forced uint32) []groupDevice {
	t.Helper()
	var out []groupDevice
	for _, d := range backend.Open() {
		if d.API() == "spirv" {
			d.Close() // reopened by index below
			continue
		}
		out = append(out, groupDevice{d.API(), d, 0})
	}
	infos, err := backend.VulkanDevices()
	if err != nil {
		t.Logf("no Vulkan: %v", err)
	}
	for _, in := range infos {
		if in.Software || !in.Compute || !testdevices.Has(t, "vulkan", in.Index) {
			continue
		}
		for _, lim := range []uint32{0, forced} {
			d, err := backend.OpenVulkanWith(strconv.Itoa(in.Index),
				backend.Opts{Vulkan: vulkan.Config{MaxGroups: lim}})
			if err != nil {
				t.Fatalf("vulkan:%d %s: %v", in.Index, in.Name, err)
			}
			name := fmt.Sprintf("vulkan%d", in.Index)
			if lim > 0 {
				name += fmt.Sprintf("-max%d", lim)
			}
			out = append(out, groupDevice{name, d, lim})
		}
	}
	if len(out) == 0 {
		t.Skip("no GPU backend on this host")
	}
	t.Cleanup(func() {
		for _, g := range out {
			g.d.Close()
		}
	})
	return out
}

// groupIDKernel writes, for every thread of every workgroup, the workgroup's
// CTAID and a value its neighbour put in workgroup memory across a barrier:
//
//	out[2f]   = ctaid                              f = ctaid*width + tid
//	out[2f+1] = ctaid*width + (tid+1) mod width    read from shared memory
//
// It does not clamp: the launch is exactly groups x width threads, so a CTAID
// that repeats leaves some pair unwritten (still poison) and one that is wrong
// writes the wrong pair. The shared-memory half is a workgroup's own state
// keyed on its identity, which is what an in-group reduction relies on.
func groupIDKernel(width int) *ir.Kernel {
	b := ir.New("groupid", [3]int{width, 1, 1})
	pOut := b.Param("pOut", ir.U32)
	sh := b.Shared("ring", ir.U32, width)
	tid, cta := b.TID(), b.CTAID()
	w := b.Const(ir.U32, int64(width))
	flat := b.Add(ir.U32, b.Mul(ir.U32, cta, b.NTID()), tid)
	b.Store(sh, tid, flat, 0)
	b.Barrier()
	next := b.Rem(ir.U32, b.Add(ir.U32, tid, b.Const(ir.U32, 1)), w)
	nb := b.Load(ir.U32, sh, next, 0)
	at := b.Mul(ir.U32, flat, b.Const(ir.U32, 2))
	b.Store(pOut, at, cta, 0)
	b.Store(pOut, at, nb, 1)
	return b.Done()
}

// TestALaunchPastTheGroupLimitRunsWhole launches more workgroups than one
// dispatch may carry and checks every thread of every workgroup wrote its own
// pair, exactly once, with poison intact after the last. It runs through both
// launch paths (a one-shot Kernel.Launch and a Session's recorded batch), at a
// one-thread and a 64-thread workgroup, on every device: at the device's own
// limit with launches past the Vulkan floor of 65535, up to 2^20 workgroups --
// which split on an integrated GPU and are one dispatch elsewhere -- and on every
// Vulkan device
// again at a forced limit of 97, so the split runs where the hardware never
// asks for it.
//
// The dispatch count is the selection check: a split launch and one that fits
// leave the same buffer, so a gate that only read the buffer could not say the
// split ran.
func TestALaunchPastTheGroupLimitRunsWhole(t *testing.T) {
	gpuLock(t)
	split, vk := 0, false
	for _, g := range groupDevices(t, forcedGroups) {
		vk = vk || g.forced > 0
		var launches [][2]int // (width, groups)
		if g.forced > 0 {
			for _, w := range []int{1, 64} {
				for _, n := range []int{1, forcedGroups / 2, forcedGroups, 2 * forcedGroups, 3*forcedGroups + 5} {
					launches = append(launches, [2]int{w, n})
				}
			}
		} else {
			// The last is sixteen times the floor: on CUDA and Metal, whose
			// limits are far above it, one dispatch of it is the evidence
			// that they need no split.
			launches = [][2]int{{1, 2*specGroups + 13}, {64, specGroups + 301}, {1, 1 << 20}}
		}
		for _, l := range launches {
			for _, session := range []bool{false, true} {
				name := fmt.Sprintf("%s/w%d/g%d/session=%v", g.name, l[0], l[1], session)
				t.Run(name, func(t *testing.T) {
					if groupIDCase(t, g.d, l[0], l[1], session) {
						split++
					}
				})
			}
		}
	}
	if !vk {
		// Only Vulkan splits; CUDA and Metal ran the launches above whole.
		t.Log("no Vulkan device here: the split path was not exercised on this host")
		return
	}
	if split == 0 {
		t.Fatal("no launch was split on any device: the split path never ran, so this gate proved nothing")
	}
	t.Logf("%d launches ran as more than one dispatch", split)
}

// groupIDCase runs groupIDKernel over groups workgroups of width threads and
// reports whether the launch was split into more than one dispatch.
func groupIDCase(t *testing.T, d backend.Device, width, groups int, session bool) bool {
	t.Helper()
	const poison, guard = 0xDEADBEEF, 0x5EED5EED
	n := 2 * groups * width
	buf := make([]byte, 4*(n+1))
	for i := 0; i < n; i++ {
		binary.LittleEndian.PutUint32(buf[4*i:], poison)
	}
	binary.LittleEndian.PutUint32(buf[4*n:], guard)
	k := groupIDKernel(width)
	if err := k.Validate(); err != nil {
		t.Fatal(err)
	}
	kk, err := d.Compile(k)
	if err != nil {
		t.Fatalf("compile: %v", err)
	}
	defer kk.Close()
	out, err := d.Alloc(len(buf))
	if err != nil {
		t.Fatalf("alloc: %v", err)
	}
	defer out.Free()
	if err := out.Write(buf); err != nil {
		t.Fatal(err)
	}
	limit, before, isVK := backend.VulkanGroups(d)
	if session {
		d.Session(func(s backend.Session) {
			err = s.Launch(kk, groups, width, out)
			if err == nil {
				err = s.Sync()
			}
		})
	} else {
		err = kk.Launch(groups, width, out)
	}
	if err != nil {
		t.Fatalf("launch: %v", err)
	}
	got := make([]byte, len(buf))
	if err := out.Read(got); err != nil {
		t.Fatal(err)
	}
	bad := 0
	for i := 0; i <= n; i++ {
		want := uint32(guard)
		if i < n {
			f := i / 2
			cta, tid := f/width, f%width
			want = uint32(cta)
			if i%2 == 1 {
				want = uint32(cta*width + (tid+1)%width)
			}
		}
		if v := binary.LittleEndian.Uint32(got[4*i:]); v != want {
			if bad++; bad <= 3 {
				t.Errorf("word %d (workgroup %d of %d) = %#x, want %#x", i, i/2/width, groups, v, want)
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d of %d words wrong", bad, n+1)
	}
	if !isVK {
		return false
	}
	_, after, _ := backend.VulkanGroups(d)
	pieces := (uint64(groups) + uint64(limit) - 1) / uint64(limit)
	if after-before != pieces {
		t.Errorf("%d workgroups at a limit of %d were %d dispatch(es), want %d",
			groups, limit, after-before, pieces)
	}
	return pieces > 1
}

// TestRealKernelsPastTheGroupLimit puts shipped kernel families through a split
// launch: Restride at a size that is past 65535 workgroups by itself (a
// transposed K of 1024 rows growing to 8448 positions, the KV resize that
// meets an integrated GPU's limit first), and at the forced limit the Restride
// shapes of TestRestrideMovesExactlyTheBlock and the matvec in its three
// launch shapes -- a k-split with its Reduce launch, the in-group split that
// reduces in workgroup memory behind a barrier, and a row tile.
func TestRealKernelsPastTheGroupLimit(t *testing.T) {
	gpuLock(t)
	const tinyGroups = 7 // the matvec shapes here launch tens of workgroups
	for _, g := range groupDevices(t, tinyGroups) {
		t.Run(g.name, func(t *testing.T) {
			if g.forced == 0 {
				rows, cols, ds := 1024, 8400, 8448
				if groups := (rows*cols + 127) / 128; groups <= specGroups {
					t.Fatalf("the restride is %d workgroups, not past %d", groups, specGroups)
				}
				_, before, isVK := backend.VulkanGroups(g.d)
				restrideCase(t, g.d, "restride/natural", rows, cols, cols, ds)
				if limit, after, _ := backend.VulkanGroups(g.d); isVK && limit < specGroups+1 && after-before < 2 {
					t.Fatalf("a restride past %d workgroups on a device limited to %d was %d dispatch(es)",
						specGroups, limit, after-before)
				}
				return
			}
			_, before, _ := backend.VulkanGroups(g.d)
			for _, c := range [][4]int{
				{1024, 257, 257, 513},
				{64, 100, 513, 257},
				{3, 7, 9, 11},
				{5, 129, 130, 200},
			} {
				restrideCase(t, g.d, fmt.Sprintf("restride/%dx%d", c[0], c[1]), c[0], c[1], c[2], c[3])
			}
			t.Run("matvec/split4", func(t *testing.T) {
				matvecCaseFull(t, g.d, kernels.Q4_K, 2048, 2048, 4, true, 1, false)
			})
			t.Run("matvec/groupsplit16", func(t *testing.T) {
				if !mainkernels.GroupSplitOK(2048, 16) {
					t.Fatal("2048 rows at split 16 is not an in-group split")
				}
				matvecCaseFull(t, g.d, kernels.Q4_K, 2048, 2048, 16, true, 1, true)
			})
			t.Run("matvec/rowt2", func(t *testing.T) {
				matvecCaseFull(t, g.d, kernels.Q4_K, 2048, 2048, 1, false, 2, false)
			})
			if _, after, _ := backend.VulkanGroups(g.d); after-before < 10 {
				t.Fatalf("%d dispatches across every case at a limit of %d: the split was not taken",
					after-before, tinyGroups)
			}
		})
	}
}
