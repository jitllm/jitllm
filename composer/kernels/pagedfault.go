//go:build jitllmfault

package kernels

// pagedFault names a deliberate error in the paged kernels' generation, so a
// gate can run each violation and demand it fails (RULE 10). Compiled in only
// under the jitllmfault tag: no release build can generate a wrong kernel.
//
//	"(t-1)/P"    the page index of position t is (t-1)/P: the first slot of
//	             every page reads the page before
//	"unaligned"  key tiles start at keyStart, not a multiple of their width,
//	             while the page id is still read once per tile
//	"normpart"   split partials are written normalised
//	"sinkchunk"  every split adds the sink to its own denominator
//	"vmask"      the prefill kernels read V without masking the positions
//	             outside a query tile's keys (a page's unwritten slots)
//	"vgroup"     FlashAttention weights every key of a V load group by the
//	             group's first key's probability
//	"vecv"       FlashDecodeKV's vector V loads credit each component to the
//	             mirrored dimension
//	"f16k"       binary16 K words read high half first
//	"f16krope"   the NEOX binary16 K rotation swaps a word's halves
//
// "unaligned" also moves the prefill kernels' partition base and their flash
// key tiles to keyStart itself.
var pagedFault string

// SetPagedFault arms one violation for the kernels generated after it; "" disarms.
func SetPagedFault(f string) { pagedFault = f }
