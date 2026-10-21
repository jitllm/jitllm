package cuda

import (
	"fmt"

	"github.com/jitllm/jitllm/jit_old/gpu/ffi"
)

// The peer calls, bound only if the driver has all three: a copy between two
// contexts, and the access that lets it go card to card. Without access
// cuMemcpyPeer still copies, staged through the host by the driver (the CUDA
// Driver API reference, cuMemcpyPeer and cuCtxEnablePeerAccess).
var (
	cuMemcpyPeer          func(CUdevptr, CUctx, CUdevptr, CUctx, uint64) CUresult
	cuDeviceCanAccessPeer func(*int32, CUdevice, CUdevice) CUresult
	cuCtxEnablePeerAccess func(CUctx, uint32) CUresult
)

// errPeerAlreadyEnabled is CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED: a second
// enable of the same pair, which is success here.
const errPeerAlreadyEnabled CUresult = 704

func loadPeer(lib *ffi.Lib) {
	for _, n := range []string{"cuMemcpyPeer", "cuDeviceCanAccessPeer", "cuCtxEnablePeerAccess"} {
		if !lib.Has(n) {
			return
		}
	}
	cuMemcpyPeer = ffi.Fn5[CUresult, CUdevptr, CUctx, CUdevptr, CUctx, uint64](lib, "cuMemcpyPeer")
	cuDeviceCanAccessPeer = ffi.Fn3[CUresult, *int32, CUdevice, CUdevice](lib, "cuDeviceCanAccessPeer")
	cuCtxEnablePeerAccess = ffi.Fn2[CUresult, CUctx, uint32](lib, "cuCtxEnablePeerAccess")
}

// PeerCopyAvailable reports whether this driver copies between contexts.
func PeerCopyAvailable() bool { return cuMemcpyPeer != nil }

// CanAccessPeer reports whether d can address peer's memory directly (NVLink
// or PCIe peer-to-peer), as the driver says.
func (d *Device) CanAccessPeer(peer *Device) bool {
	if cuDeviceCanAccessPeer == nil || d == peer {
		return false
	}
	var can int32
	if cuDeviceCanAccessPeer(&can, d.dev, peer.dev) != 0 {
		return false
	}
	return can != 0
}

// EnablePeerAccess lets the context current on the calling thread -- d's,
// on its owner thread -- address peer's allocations. Run it on d's thread.
func (d *Device) EnablePeerAccess(peer *Device) error {
	if cuCtxEnablePeerAccess == nil {
		return fmt.Errorf("cuda: this driver has no cuCtxEnablePeerAccess")
	}
	if r := cuCtxEnablePeerAccess(peer.ctx, 0); r != 0 && r != errPeerAlreadyEnabled {
		return call(r, "cuCtxEnablePeerAccess")
	}
	return nil
}

// CopyPeer copies n bytes of src (an allocation of srcDev) at srcOff to dst
// (dstDev's) at dstOff. It returns once the copy is done with respect to the
// host (cuMemcpyPeer's synchronous behaviour). Ranges are the caller's to have
// checked.
func CopyPeer(dst *Buffer, dstDev *Device, dstOff int, src *Buffer, srcDev *Device, srcOff, n int) error {
	if cuMemcpyPeer == nil {
		return fmt.Errorf("cuda: this driver has no cuMemcpyPeer")
	}
	return call(cuMemcpyPeer(dst.p+CUdevptr(dstOff), dstDev.ctx, src.p+CUdevptr(srcOff), srcDev.ctx, uint64(n)), "cuMemcpyPeer")
}
