package gpu

import (
	"encoding/binary"

	"github.com/jitllm/jitllm/backend"
	"github.com/jitllm/jitllm/composer"
	"github.com/jitllm/jitllm/composer/kernels"
)

// entDesc is a call's rows' entry descriptors at one rate a block folds its
// entries at -- a row's run in the table arena, the entries it reads, the
// one it writes -- and their buffer. A row completing no entry writes the
// last entry its run can name, past every entry a row reads.
type entDesc struct {
	rate  int
	words []uint32
	buf   backend.Buf
}

// initEntDescs gives pg a descriptor set for each rate in rates, for a
// scratch of rows rows. Callers hold g.mu.
func (g *devTier) initEntDescs(pg *pagedScratch, rows int, rates ...int) error {
	for _, r := range rates {
		if r <= 0 {
			continue
		}
		b, err := g.dev.Alloc(rows * kernels.PRowWords * 4)
		if err != nil {
			return err
		}
		pg.ents = append(pg.ents, entDesc{rate: r, buf: b})
	}
	return nil
}

// stageEnts fills pg's entry descriptor sets for a call's rows.
func (pg *pagedScratch) stageEnts(kp *kvPool, rows []pagedRow) {
	for i := range pg.ents {
		e := &pg.ents[i]
		e.words = e.words[:0]
		for _, r := range rows {
			if r.pad {
				e.words = append(e.words, 0, 0, 0, 0)
				continue
			}
			rg := kp.ranges[r.s]
			w := uint32(rg.cap*kp.p - 1)
			if (r.pos+1)%e.rate == 0 {
				w = uint32(r.pos / e.rate)
			}
			e.words = append(e.words, uint32(rg.off), 0, uint32((r.pos+1)/e.rate), w)
		}
	}
}

// entRows is pg's entry descriptors at rate, nil where it stages none.
func (pg *pagedScratch) entRows(rate int) backend.Buf {
	if pg == nil {
		return nil
	}
	for i := range pg.ents {
		if pg.ents[i].rate == rate {
			return pg.ents[i].buf
		}
	}
	return nil
}

// bindEnts binds block li's entries into t where its graph keeps them: the
// entry pool's keys and table, and bs's descriptors at the block's rate; and
// the rows' tokens where it routes by them.
func (g *devTier) bindEnts(t map[composer.TableKey]backend.Buf, l *layer, li int, bs *blockScratch) {
	if bs != nil && bs.rowTok != nil {
		t[tableKey(composer.TableTokens)] = bs.rowTok
	}
	if bs != nil && bs.rowPos != nil {
		t[tableKey(composer.TablePos)] = bs.rowPos
	}
	if g.kvp == nil || bs == nil {
		return
	}
	ents := bs.pkv
	el := g.kvp.layers[entKey(li)]
	if el == nil {
		return
	}
	t[tableKey(composer.TableEntKeys)], t[tableKey(composer.TableEntPages)], t[tableKey(composer.TableEntRows)] = el.k, el.tab, ents.entRows(el.rate)
}

// tokWords is the call's R rows' tokens (SetTokenIDs) as words, a row past
// the ids the last one's.
func (g *devTier) tokWords(R int) []byte {
	b := g.subBytes.rowTok[:0]
	n := len(g.tokIDs)
	for r := range R {
		var id int32 = -1
		if n > 0 {
			id = g.tokIDs[min(r, n-1)]
		}
		b = binary.LittleEndian.AppendUint32(b, uint32(id))
	}
	g.subBytes.rowTok = b
	return b
}
