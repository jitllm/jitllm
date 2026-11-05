# Paging is the instance's

Stated by the user: a model has several instances, each paging -- if it pages
-- a different part of the model. AGENTS.md's shape says the same of an
instance: "every kernel for one placement (which device each block runs on,
**how heavily it pages**)". This file is the design; the rule is AGENTS.md's.

## What is wrong now

The container is the file and the pager in one (`jlm.File`): one page array
(`pages`), one budget (`SetBudget`), one resident set, its free lists, the
dense region and the expert pages' reservation. `Model.SetPageBudget` sets that
one budget (and re-binds every block); the server and `cmd/jitllm` set it once
per model. The model's block weights are bound into its frames (`bindPacked`
re-points `model.Layer`'s tensors at a page-in), and `Model.CpuUse` counts the
instances running a block on the CPU so one does not unbind what another runs
(`Model.Bind`).

So:

- an instance has no budget and no resident set of its own: one instance's
  page-in evicts a block another is about to run, and MRU over the cyclic
  scan (`sched/cyclic.go`) is computed over two scans interleaved;
- the model holds the engine's devices, because its pager is the only pager:
  the cards told of a recycled frame (`HeldBy`, `ForgetCopies`), the devices
  released at its Close (`nn.ModelOwner`), the host-memory devices netted off
  its budget (`HoldOnHost`).

## The shape

**The model** is the weights as a file: the container's header, its page
table (every block's, expert's and vision block's offset and size), the
config, the graphs, and the layer descriptors (roles, shapes, where a tensor
lies inside its page). It reads a page or a page's ranges into memory it is
handed (O_DIRECT, as now). It holds no frame -- the dense region included --
no budget, no resident set, no device and no callback.

**Every device pages the blocks placed on it, the same way.** The CPU and a
card are alike: a device reads the pages of the blocks an instance places on
it from the file into memory of its own, keeps them under that instance's
quota on it, evicts in that instance's order, and holds a block two
instances place on it once. One contract, every device implementing it; no
device pages through another's memory, and there is no ledger beside the
devices. The File only reads into memory it is handed.

**An instance's budget is a retention quota.** It bounds the bytes that
instance keeps pinned between uses, shared or not, and its retention order is
MRU over its own scan (LRU for expert pages) -- an instance's policy, not a
claim of global optimality over a shared cache. Unpinned frames stay as the
ledger's clean cache under its own deterministic order; physical pressure
reclaims those first, then asks instances to relinquish, and never recycles a
pinned frame. Admission is at placement, as a KV reservation is: an instance
whose working set (the largest run of blocks it must hold at once) does not
fit what the ledger can give waits or is refused there, never mid-token.

**One lifetime protocol for a frame**: acquire (a hit pins; a miss reserves a
frame, the reads fill it, publishing makes it readable), leases while a block
executes or transfers (a lease outlives an unpin), and a generation that
moves when a frame is reused -- a card's copy keyed on a host address keys on
the generation too, so a reused frame invalidates it without a callback
registered anywhere. A device's hold of a model's blocks is reference counted
by the instances placing them: the last instance's release gives them back,
never one instance's Close while another still runs them.

**An instance's weights point into the frames it leased**: its own bindings
(the CPU form's roles, the `nn.LayerWeights` it offers a card), not
`model.Layer`'s.

The rules that stand: no mmap of files (frames are anonymous memory, pages read
with O_DIRECT); every expert its own page; paging is always on (a model that
fits is an instance whose quota holds every page it runs); a warm token and a
warm fault allocate nothing (the ledger's bookkeeping is preallocated).

## Steps, each its own commit and gates

1. **One pager and binding object the engine goes through.** Every page-in,
   ensure, bind, budget and recycle the engine asks of the model goes through
   one engine object over today's model-wide pager, with today's behaviour
   kept: `Model.PageIn`, `PageInSelected`, `EnsureBlock*`, `SetPageBudget`,
   `ForgetCopies`, `HeldBy`, `HoldOnHost` lose their engine callers. Gates:
   the paging principle's (`TestPagingDoesNotChangeTheAnswer`,
   `TestOpenHonoursAPageBudget`, `TestForwardBatchFaultsItsBlocksIn`,
   `TestTwoPagedModels*`, `TestWindowedLayersHoldTheirWindow`,
   `TestBlocksPageOnTheListedCard`), the expert paging gates,
   `TestDecodeDoesNotAllocate`.
2. **Bindings to the instance.** `model.Layer` loses its addresses; an
   instance binds from the frames it leased. `Model.CpuUse`, `Model.Bind` and
   the re-bind in `SetPageBudget` go. Gates: as 1, plus the relocation
   principle's (a block moving off and back re-binds).
3. **Frames out of the file**, in two commits.
   - **3a**: `jlm.Frames` is split from `jlm.File`. The File keeps the
     header, the table, the layout (page bounds, extents) and the reads into
     memory it is handed; Frames hold the frames, the dense region, the
     budget and resident set, the expert reservation, the read service,
     preload and speculation, over a File (`jlm.NewFrames`). The model still
     opens both (`Model.Frames`); every caller of a frame method is
     re-pointed at the frames.
   - **3b**: each device pages for itself, through one contract every
     device implements: the CPU's into its memory, a card's into its own,
     each reading the file directly -- a card stops reading through the
     CPU's frames. The model opens the File alone and reads its vectors from
     it at open; an instance binds its own copies of its blocks and dense
     tensors at the pages the device running them holds. Address-keyed
     copies key on the page's generation; `OnRecycle` goes. (A shared
     engine-level ledger, and the same pager filed under the CPU device,
     were both taken out: each was the model-wide pager under another
     owner, and the second treated the CPU unlike a card.)
   Gates: as 2, plus the vision and expert gates.
4. **Quotas per instance.** `engine.WithPageBudget`; `Model.SetPageBudget`
   goes; the server's per-model budget becomes each instance's, the
   `SetPageBudget` RPC names the instance (a proto change, buf generate);
   admission at placement. Gates: as 3, and the server's.
5. **The model holds nothing of a device.** Its registries go; `srcgate` holds
   that no type of `model/` has a field of an `nn` device interface or a
   func, and that `Model.Close` reaches no device.
6. **The transitions, gated.** Two instances of one model -- one at half its
   pages on the CPU, one with a seam on a card -- each matching the
   reference, the blocks both run held once (the ledger's frame count), with
   overlapping scans; the last owner closing while the other runs; a quota
   shrinking during a read; expert, vision and dense pages covered; a warm
   fault and re-bind allocating nothing; cards-only paging; every leasing
   session's history relocating with its blocks. The violation is the
   model-wide pager (step 1's tree).

## Open

- Admission's two modes: a placement that guarantees independent progress
  reserves its working set; one that relies on sharing reserves only what is
  not already pinned, and says so.
- `WithPreload` reads every page at open: it becomes a warm of the ledger's
  clean cache, nothing pinned until an instance pins it.
