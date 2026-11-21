package metal

import (
	"fmt"
	"math"
)

// launchable refuses a dispatch whose threadgroup indices the kernel cannot
// hold.
//
// Metal reports no limit on the threadgroups of one compute dispatch -- an
// MTLDevice answers for threads per threadgroup, not per grid -- and takes the
// count as an NSUInteger. What bounds it is the kernel: msl.Emit reads the
// IR's CTAID from [[threadgroup_position_in_grid]] as a 32-bit uint, so a grid
// past 2^32 groups would wrap and run two threadgroups as one item. Below that
// a launch runs whole, which is why this backend, unlike Vulkan, never splits
// one: there is no device limit at 65535 or anywhere under the index's reach.
func launchable(groups int) error {
	if groups < 0 || uint64(groups) > math.MaxUint32 {
		return fmt.Errorf("metal: dispatch of %d threadgroups; the kernel's threadgroup index is 32-bit, "+
			"so a dispatch holds at most %d", groups, uint64(math.MaxUint32))
	}
	return nil
}
