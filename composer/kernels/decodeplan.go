package kernels

import "math"

// Choosing a paged decode's attention: which path, how many key partitions,
// how wide a workgroup and which merge. These are pure functions of the
// shape, the key count a compiled variant must cover and the device's
// resident-thread count (Device.Slots; 0 when the backend cannot say), so the
// tier can call them wherever it compiles a variant and regenerate when the
// answer changes. The measurements behind every rule are in
// docs/engineering-history/gpu-kernels.md, "Faster paged decode attention".

// DecodePath is the kernel family a paged decode runs.
type DecodePath int

const (
	// PathFlashKV is FlashDecodeKV, then FlashAttentionMergeWide when split.
	PathFlashKV DecodePath = iota
	// PathStaged is PagedStagedScores, PagedStagedSoftmax, PagedStagedAcc and
	// FlashAttentionMergeWide.
	PathStaged
)

func (p DecodePath) String() string {
	if p == PathStaged {
		return "staged"
	}
	return "flashkv"
}

// DecodePlan is one paged decode attention: the path and the shape its
// kernels are generated for (Splits, Warps, Chunk set).
type DecodePlan struct {
	Path  DecodePath
	Shape FlashShape
}

// slotsOr16K is slots, or 16384 where the device does not report it.
func slotsOr16K(slots int) int {
	if slots <= 0 {
		return 16384
	}
	return slots
}

// FlashKVPlan is s (a paged FlashDecodeKV shape, one row or several) with
// its warps and split count chosen for keys keys:
//
//   - A head wider than 128 at 4096 keys or more takes 4 warps rather than
//     one per 32 dimensions. The narrower group holds a query head's slice
//     in fewer lanes, so flashKVGroup keeps one head a workgroup and the
//     chunk grows from 256 keys to 1920: the split count the chunk forces
//     falls 4-7x. It pays from 4096 keys and loses at 512.
//   - Splits are tier/flash.go's rule -- the power of two nearest
//     sqrt(keys/8), capped at ~1.6 workgroups an SM, floored by what the
//     chunk needs -- with the cap rounded UP. Rounded down, a 20-SM part with
//     16 workgroups at one split got a cap of 1 and lost 2 to 8 splits at
//     512 keys.
//
// The merge for its partials is FlashAttentionMergeWide, never slower than
// the group-per-item merge.
func FlashKVPlan(s FlashShape, keys, slots int) FlashShape {
	slots = slotsOr16K(slots)
	if s.Warps == 0 && s.Dim > 128 && keys >= 4096 {
		s.Warps = 4
	}
	s.Splits = 1
	groups := FlashKVGroups(s)
	floor := (keys + FlashKVChunk(s) - 1) / FlashKVChunk(s)
	capped := max(1, (slots+1280*groups-1)/(1280*groups))
	sq := 1 << int(math.Round(math.Log2(math.Sqrt(max(1, float64(keys)/8)))))
	s.Splits = max(1, floor, min(sq, capped))
	return s
}

// PagedStagedPlan is s (a paged staged shape) with its split count and chunk
// chosen for keys keys: the power of two at or below sqrt(keys/8), raised
// until the accumulate -- Rows*Heads/G*Splits*Dim threads, G falling as the
// chunk shrinks (stagedGroup) -- launches a quarter of the device's resident
// threads, and no chunk under 32 keys. sqrt(keys/8) alone starved a large
// device: 16 splits at 4096 keys launched 8192 accumulate threads on 80 SMs,
// and 64 splits ran about twice as fast; at 16384, 128 splits beat 32 by
// about 1.8-2x. Past the square root is a loss where the device is already
// full.
//
// keys is the longest span a row's partition covers: keyEnd less keyStart
// rounded down to 32.
func PagedStagedPlan(s FlashShape, keys, slots int) FlashShape {
	slots = slotsOr16K(slots)
	keys = max(1, keys)
	sp := 1 << int(math.Log2(math.Sqrt(max(1, float64(keys)/8))))
	for ; ; sp *= 2 {
		t := PagedStagedAt(s, keys, sp)
		if 4*PagedStagedAccThreads(t) >= slots || t.Chunk == 32 {
			return t
		}
	}
}

// PagedStagedAt is s at splits splits over keys keys: the chunk is the keys a
// split covers, rounded up to 32.
func PagedStagedAt(s FlashShape, keys, splits int) FlashShape {
	s.Splits = splits
	s.Chunk = max(32, ((max(1, keys)+splits-1)/splits+31)/32*32)
	return s
}

// ChooseDecodePlan is the paged decode attention for keys keys on a device
// of the given API ("ptx", "spirv", "msl") and resident-thread count: the
// path, and its shape from FlashKVPlan or PagedStagedPlan.
//
// The staged path wins where a KV head's history is large against the
// device: on PTX when keys*Dim reaches 24 resident threads' worth of the
// device. On a large part FlashDecodeKV still wins at 4096 keys. NVIDIA's
// Vulkan driver gives FlashDecodeKV the shorter histories by more and the
// staged path 16384 keys and up. Metal follows PTX's rule at its fallback
// of 16384 threads. Four warps past 128 dimensions is a loss on the Vulkan
// driver at 4096 keys and a win at 16384, so there it waits for 16384; on
// Metal it is not taken.
//
// A tier that can time both paths on its own pools should: the crossover
// moves by a factor of four between two NVIDIA parts and by API on one.
func ChooseDecodePlan(api string, s FlashShape, keys, slots int) DecodePlan {
	staged := false
	switch api {
	case "ptx", "msl":
		staged = keys*s.Dim >= 24*slotsOr16K(slots)
	case "spirv":
		staged = keys >= 16384
	}
	if s.MLA > 0 {
		staged = true // FlashDecodeKV has no MLA layout
	}
	if staged {
		return DecodePlan{Path: PathStaged, Shape: PagedStagedPlan(s, keys, slots)}
	}
	kv := s
	if kv.Warps == 0 && ((api == "spirv" && keys < 16384) || api == "msl") {
		kv.Warps = flashKVWarps(kv) // the default width: no four-warp rule
	}
	return DecodePlan{Path: PathFlashKV, Shape: FlashKVPlan(kv, keys, slots)}
}
