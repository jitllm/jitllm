// Package view draws jivman with gg: the app bar, the board, the HUD, the
// decision panel, the leaderboard and the Play dialog, after jevman's page
// (its palette, layout and board), recreated from primitives. It draws into
// any gg.Context: the window's canvas, or an offscreen one for a screenshot.
package view

import (
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/gogpu/gg"
	"github.com/gogpu/gg/text"
	"golang.org/x/image/font/gofont/gobold"
	"golang.org/x/image/font/gofont/gomono"
	"golang.org/x/image/font/gofont/gomonobold"
	"golang.org/x/image/font/gofont/goregular"

	"github.com/jitllm/jitllm/examples/jivman/game"
	"github.com/jitllm/jitllm/examples/jivman/play"
	"github.com/jitllm/jitllm/examples/jivman/scores"
)

// The page is drawn at this size; the window is made this size.
const (
	W    = 1200
	H    = 800
	Tile = 20

	boardX = 40
	boardY = 100
	panelX = 640
	panelY = 76
	panelW = 520
	panelH = 700
)

// jevman's palette (its style.css and canvas renderer).
var (
	cBg      = gg.Hex("#0b0b12")
	cPanel   = gg.Hex("#14141f")
	cCard    = gg.Hex("#1c1c2b")
	cText    = gg.Hex("#e8e8f0")
	cMuted   = gg.Hex("#8a8aa0")
	cAccent  = gg.Hex("#ffd800")
	cDanger  = gg.Hex("#ff5c5c")
	cBorder  = gg.Hex("#22223a")
	cButton  = gg.Hex("#24243a")
	cButtonB = gg.Hex("#3a3a5a")
	cWall    = gg.Hex("#2121de")
	cDoor    = gg.Hex("#ffb8ff")
	cPellet  = gg.Hex("#ffb8ae")
	cFright  = gg.Hex("#2121ff")
	cBlack   = gg.Hex("#000000")
	cWhite   = gg.Hex("#ffffff")
	cCyan    = gg.Hex("#00ffff")
	// Actors is Pac-Man's yellow and the ghosts' arcade colours.
	Actors = [game.NActors]gg.RGBA{gg.Hex("#ffd800"), gg.Hex("#ff0000"), gg.Hex("#ffb8ff"), gg.Hex("#00ffff"), gg.Hex("#ffb852")}
)

var arrows = [4]string{"↑", "←", "↓", "→"}

// Fonts are the Go fonts (BSD licensed, shipped in golang.org/x/image).
type Fonts struct {
	sans, bold, mono, monoBold *text.FontSource
	faces                      map[faceKey]text.Face
}

type faceKey struct {
	src  *text.FontSource
	size float64
}

// LoadFonts parses the fonts once.
func LoadFonts() (*Fonts, error) {
	f := &Fonts{faces: map[faceKey]text.Face{}}
	for _, x := range []struct {
		dst  **text.FontSource
		data []byte
	}{{&f.sans, goregular.TTF}, {&f.bold, gobold.TTF}, {&f.mono, gomono.TTF}, {&f.monoBold, gomonobold.TTF}} {
		s, err := text.NewFontSource(x.data)
		if err != nil {
			return nil, err
		}
		*x.dst = s
	}
	return f, nil
}

func (f *Fonts) face(src *text.FontSource, size float64) text.Face {
	k := faceKey{src, size}
	if fc, ok := f.faces[k]; ok {
		return fc
	}
	fc := src.Face(size)
	f.faces[k] = fc
	return fc
}

// Tab is the page shown beside the board.
type Tab uint8

// The tabs.
const (
	TabPlay Tab = iota
	TabBoard
)

// Scene is everything one frame shows.
type Scene struct {
	G     *game.Game
	Stats *play.Stats
	// PacmanBy and GhostsBy name who plays each side: a model's name, "You"
	// or "Classic".
	PacmanBy, GhostsBy string
	Device             string
	Board              []scores.Row
	High               int
	Tab                Tab
	Paused             bool
	// Thinking marks the actors stopped at a junction for their answer.
	Thinking [game.NActors]bool
	// Driven marks the actors a model plays.
	Driven [game.NActors]bool
	T      float64
	// Summary is the finished run, shown in the panel.
	Summary string

	Dialog *Dialog
	// Banner is a line across the top of the board: what a recording's
	// clock is, when it is not the wall's.
	Banner string

	hits []Hit
}

// Dialog is the Play dialog: who plays Pac-Man, who plays the ghosts.
type Dialog struct {
	Models []string
	// Pacman and Ghosts are an index into Models, or -1 for You / Classic.
	Pacman, Ghosts int
	Status         string
	Busy           bool
}

// Action is what a click asks for.
type Action struct {
	Kind  ActionKind
	Index int
}

// ActionKind names a clickable thing.
type ActionKind uint8

// The actions.
const (
	ActNone ActionKind = iota
	ActOpenPlay
	ActPickPacman
	ActPickGhosts
	ActStart
	ActCloseDialog
	ActTab
	ActPause
	ActRestart
	ActTakeOver
)

// Hit is a clickable rectangle of the last frame.
type Hit struct {
	X, Y, W, H float64
	Action
}

// At is the action under (x, y) in the last frame drawn, topmost first.
func (s *Scene) At(x, y float64) Action {
	for i := len(s.hits) - 1; i >= 0; i-- {
		h := s.hits[i]
		if x >= h.X && x < h.X+h.W && y >= h.Y && y < h.Y+h.H {
			return h.Action
		}
	}
	return Action{}
}

func (s *Scene) hit(x, y, w, h float64, a Action) { s.hits = append(s.hits, Hit{x, y, w, h, a}) }

// Painter draws scenes.
type Painter struct {
	F *Fonts
}

func fill(dc *gg.Context, c gg.RGBA) {
	dc.SetRGBA(c.R, c.G, c.B, c.A)
	dc.Fill()
}

func alpha(c gg.RGBA, a float64) gg.RGBA { c.A = a; return c }

func (p *Painter) rect(dc *gg.Context, x, y, w, h, r float64, c gg.RGBA) {
	if r > 0 {
		dc.DrawRoundedRectangle(x, y, w, h, r)
	} else {
		dc.DrawRectangle(x, y, w, h)
	}
	fill(dc, c)
}

func (p *Painter) text(dc *gg.Context, s string, x, y float64, src *text.FontSource, size float64, c gg.RGBA, ax float64) float64 {
	dc.SetFont(p.F.face(src, size))
	dc.SetRGBA(c.R, c.G, c.B, c.A)
	dc.DrawStringAnchored(s, x, y, ax, 0.5)
	w, _ := dc.MeasureString(s)
	return w
}

func (p *Painter) measure(dc *gg.Context, s string, src *text.FontSource, size float64) float64 {
	dc.SetFont(p.F.face(src, size))
	w, _ := dc.MeasureString(s)
	return w
}

// Draw paints s.
func (p *Painter) Draw(dc *gg.Context, s *Scene) {
	s.hits = s.hits[:0]
	dc.ClearWithColor(cBg)
	p.appbar(dc, s)
	p.hud(dc, s)
	p.board(dc, s)
	p.controls(dc, s)
	if s.Banner != "" {
		p.rect(dc, boardX+8, boardY+4, p.measure(dc, s.Banner, p.F.sans, 12)+20, 22, 6, gg.RGBA2(0.08, 0.08, 0.12, 0.88))
		p.text(dc, s.Banner, boardX+18, boardY+15, p.F.sans, 12, cAccent, 0)
	}
	if s.Tab == TabBoard {
		p.leaderboard(dc, s)
	} else {
		p.panel(dc, s)
	}
	if s.Dialog != nil {
		p.dialog(dc, s)
	}
}

func (p *Painter) appbar(dc *gg.Context, s *Scene) {
	p.rect(dc, 0, 0, W, 56, 0, cPanel)
	p.rect(dc, 0, 56, W, 1, 0, cBorder)
	p.pacman(dc, 54, 28, 14, 0, 0.3*math.Pi, cAccent)
	p.text(dc, "jivman", 78, 20, p.F.bold, 19, cText, 0)
	p.text(dc, "Watch a decision model play Pac-Man in this process, or try to beat it", 78, 40, p.F.sans, 12, cMuted, 0)
	x := 560.0
	for i, name := range []string{"Play", "Leaderboard"} {
		w := p.measure(dc, name, p.F.bold, 14) + 24
		if Tab(i) == s.Tab {
			p.rect(dc, x, 14, w, 30, 6, cButton)
		}
		c := cMuted
		if Tab(i) == s.Tab {
			c = cText
		}
		p.text(dc, name, x+12, 29, p.F.bold, 14, c, 0)
		s.hit(x, 14, w, 30, Action{ActTab, i})
		x += w + 4
	}
	right := "jitllm"
	if s.Device != "" {
		right = "jitllm · " + s.Device
	}
	p.text(dc, right, W-40, 28, p.F.sans, 13, cMuted, 1)
}

func (p *Painter) hud(dc *gg.Context, s *Scene) {
	g := s.G
	x := float64(boardX)
	y := 86.0
	item := func(label, val string) {
		x += p.text(dc, label+" ", x, y, p.F.mono, 15, cText, 0)
		x += p.text(dc, val, x, y, p.F.monoBold, 15, cAccent, 0) + 20
	}
	item("Score", strconv.Itoa(g.Score))
	item("High", strconv.Itoa(max(s.High, g.Score)))
	item("Level", strconv.Itoa(g.Level))
	item("Lives", strconv.Itoa(max(g.Lives, 0)))
	fr := "-"
	if g.Fruit.On {
		fr = g.Fruit.Kind
	} else if k, _ := game.FruitFor(g.Level); k != "" {
		fr = k
	}
	item("Fruit", fr)
}

func (p *Painter) controls(dc *gg.Context, s *Scene) {
	y := boardY + 31*Tile + 14.0
	x := float64(boardX)
	button := func(label string, a Action, primary bool) {
		w := p.measure(dc, label, p.F.sans, 14) + 26
		bg, fg := cButton, cText
		if primary {
			bg, fg = cAccent, cBlack
		}
		dc.DrawRoundedRectangle(x, y, w, 34, 6)
		fill(dc, bg)
		if !primary {
			dc.DrawRoundedRectangle(x+0.5, y+0.5, w-1, 33, 6)
			dc.SetRGBA(cButtonB.R, cButtonB.G, cButtonB.B, 1)
			dc.SetLineWidth(1)
			dc.Stroke()
		}
		p.text(dc, label, x+13, y+17, p.F.sans, 14, fg, 0)
		s.hit(x, y, w, 34, a)
		x += w + 10
	}
	button("Play", Action{Kind: ActOpenPlay}, true)
	who := "Pac-Man: " + s.PacmanBy
	if s.PacmanBy != "You" && s.PacmanBy != "" {
		who = "Pac-Man: AI"
	}
	button(who, Action{Kind: ActTakeOver}, false)
	if s.Paused {
		button("Resume", Action{Kind: ActPause}, false)
	} else {
		button("Pause", Action{Kind: ActPause}, false)
	}
	button("Restart", Action{Kind: ActRestart}, false)
	p.text(dc, "Arrows/WASD steer when you play Pac-Man · J: Pac-Man to the AI or back · P pause · R restart",
		boardX, y+52, p.F.sans, 12, cMuted, 0)
}

// px is a tile coordinate's pixel centre.
func px(t float64) float64 { return (t + 0.5) * Tile }

func (p *Painter) board(dc *gg.Context, s *Scene) {
	g := s.G
	m := g.Maze
	dc.Push()
	ox, oy := float64(boardX), float64(boardY)
	p.rect(dc, ox, oy, float64(m.W*Tile), float64(m.H*Tile), 6, cBlack)
	isWall := func(x, y int) bool {
		if y < 0 || y >= m.H || x < 0 || x >= m.W {
			return true
		}
		return m.Rows[y][x] == '#'
	}
	dc.SetLineWidth(2)
	dc.SetRGBA(cWall.R, cWall.G, cWall.B, 1)
	for y := 0; y < m.H; y++ {
		for x := 0; x < m.W; x++ {
			c := m.Rows[y][x]
			l, r, t, b := ox+float64(x*Tile), ox+float64((x+1)*Tile), oy+float64(y*Tile), oy+float64((y+1)*Tile)
			switch c {
			case '#':
				if !isWall(x, y-1) {
					dc.MoveTo(l, t+1)
					dc.LineTo(r, t+1)
				}
				if !isWall(x, y+1) {
					dc.MoveTo(l, b-1)
					dc.LineTo(r, b-1)
				}
				if !isWall(x-1, y) {
					dc.MoveTo(l+1, t)
					dc.LineTo(l+1, b)
				}
				if !isWall(x+1, y) {
					dc.MoveTo(r-1, t)
					dc.LineTo(r-1, b)
				}
			}
		}
	}
	dc.Stroke()
	for y := 0; y < m.H; y++ {
		for x := 0; x < m.W; x++ {
			if m.Rows[y][x] == '-' {
				p.rect(dc, ox+float64(x*Tile), oy+float64(y*Tile)+Tile/2-2, Tile, 4, 0, cDoor)
			}
			k := y*m.W + x
			cx, cy := ox+px(float64(x)), oy+px(float64(y))
			if m.Pellet[k] {
				dc.DrawCircle(cx, cy, 2)
				fill(dc, cPellet)
			} else if m.Power[k] && int(s.T*4)%2 == 0 {
				dc.DrawCircle(cx, cy, 6)
				fill(dc, cPellet)
			}
		}
	}
	if g.Fruit.On {
		p.fruit(dc, ox+px(float64(g.Fruit.Tile.X)), oy+px(float64(g.Fruit.Tile.Y)), g.Fruit.Kind)
	}
	p.odds(dc, s, ox, oy)
	for id := game.Blinky; id < game.NActors; id++ {
		p.ghost(dc, s, id, ox, oy)
	}
	if g.Status != game.GameOver {
		a := &g.Actors[game.Pacman]
		x, y := a.Position()
		mouth := 0.25 * math.Pi * math.Abs(math.Sin(s.T*12))
		if g.Status == game.Dying {
			mouth = math.Min(math.Pi, (1-g.StatusTimer/game.DyingSeconds)*math.Pi)
		}
		p.pacman(dc, ox+px(x), oy+px(y), Tile*0.45, dirAngle[a.Dir], mouth, cAccent)
	}
	for id := 0; id < game.NActors; id++ {
		if s.Thinking[id] {
			x, y := g.Actors[id].Position()
			p.thinking(dc, ox+px(x), oy+px(y)-Tile, s.T)
		}
	}
	for _, pu := range g.Popups {
		if pu.TTL > 0 {
			p.text(dc, strconv.Itoa(pu.Points), ox+px(float64(pu.Tile.X)), oy+px(float64(pu.Tile.Y)), p.F.monoBold, 11, cCyan, 0.5)
		}
	}
	cx, cy := ox+px(float64(game.FruitTile.X)+0.5), oy+px(float64(game.FruitTile.Y))
	switch g.Status {
	case game.Ready:
		p.text(dc, "READY!", cx, cy, p.F.monoBold, 16, cAccent, 0.5)
	case game.GameOver:
		p.text(dc, "GAME OVER", cx, cy, p.F.monoBold, 16, gg.Hex("#ff0000"), 0.5)
		p.text(dc, "press R to restart", cx, cy+30, p.F.monoBold, 10, cWhite, 0.5)
	case game.LevelClear:
		p.text(dc, "LEVEL CLEAR", cx, cy, p.F.monoBold, 16, cWhite, 0.5)
	}
	if s.Paused {
		p.text(dc, "PAUSED", cx, cy-60, p.F.monoBold, 16, cWhite, 0.5)
	}
	dc.Pop()
}

var dirAngle = [4]float64{-math.Pi / 2, math.Pi, math.Pi / 2, 0}

// pacman draws the wedge as a polygon (gg's arc does not follow a transform).
func (p *Painter) pacman(dc *gg.Context, x, y, r, angle, mouth float64, c gg.RGBA) {
	dc.MoveTo(x, y)
	const n = 28
	a0, a1 := angle+mouth, angle+2*math.Pi-mouth
	for i := 0; i <= n; i++ {
		a := a0 + (a1-a0)*float64(i)/n
		dc.LineTo(x+r*math.Cos(a), y+r*math.Sin(a))
	}
	dc.ClosePath()
	fill(dc, c)
}

func (p *Painter) ghost(dc *gg.Context, s *Scene, id int, ox, oy float64) {
	g := s.G
	a := &g.Actors[id]
	var x, y float64
	if a.State == game.House {
		x, y = float64(game.HouseTiles[id].X), float64(game.HouseTiles[id].Y)+0.15*math.Sin(s.T*6)
	} else {
		x, y = a.Position()
	}
	cx, cy, r := ox+px(x), oy+px(y), Tile*0.45
	if a.State != game.Eaten {
		body := Actors[id]
		if a.State == game.Frightened {
			body = cFright
			if g.FrightLeft < 2 && int(s.T*5)%2 == 0 {
				body = cWhite
			}
		}
		const n = 14
		dc.MoveTo(cx-r, cy-r*0.1)
		for i := 0; i <= n; i++ {
			an := math.Pi + math.Pi*float64(i)/n
			dc.LineTo(cx+r*math.Cos(an), cy-r*0.1+r*math.Sin(an))
		}
		dc.LineTo(cx+r, cy+r)
		for i, f := range []float64{0.66, 0.33, 0, -0.33, -0.66, -1} {
			yy := cy + r
			if i%2 == 0 {
				yy = cy + r*0.6
			}
			dc.LineTo(cx+r*f, yy)
		}
		dc.ClosePath()
		fill(dc, body)
	}
	if a.State == game.Frightened {
		for _, side := range []float64{-1, 1} {
			dc.DrawCircle(cx+side*r*0.35, cy-r*0.2, r*0.12)
			fill(dc, cPellet)
		}
		return
	}
	dx, dy := a.Dir.Vec()
	for _, side := range []float64{-1, 1} {
		ex, ey := cx+side*r*0.35, cy-r*0.2
		dc.DrawCircle(ex, ey, r*0.28)
		fill(dc, cWhite)
		dc.DrawCircle(ex+float64(dx)*r*0.12, ey+float64(dy)*r*0.12, r*0.14)
		fill(dc, cFright)
	}
}

func (p *Painter) thinking(dc *gg.Context, x, y, t float64) {
	p.rect(dc, x-12, y-6, 24, 12, 6, gg.RGBA2(1, 1, 1, 0.92))
	active := int(t*4) % 3
	for i := 0; i < 3; i++ {
		dc.DrawCircle(x-6+float64(i)*6, y, 1.8)
		if i == active {
			fill(dc, cBlack)
		} else {
			fill(dc, gg.Hex("#888888"))
		}
	}
}

// fruit draws the bonus from primitives: jevman uses emoji, which the Go
// fonts do not carry.
func (p *Painter) fruit(dc *gg.Context, x, y float64, kind string) {
	switch kind {
	case "cherry":
		dc.SetLineWidth(1.5)
		dc.SetRGBA(0.3, 0.7, 0.2, 1)
		dc.MoveTo(x-3, y+2)
		dc.LineTo(x+2, y-7)
		dc.LineTo(x+5, y+3)
		dc.Stroke()
		dc.DrawCircle(x-3, y+3, 3.6)
		fill(dc, gg.Hex("#ff2020"))
		dc.DrawCircle(x+4, y+4, 3.6)
		fill(dc, gg.Hex("#ff2020"))
	case "strawberry":
		dc.MoveTo(x-6, y-3)
		dc.LineTo(x+6, y-3)
		dc.LineTo(x, y+7)
		dc.ClosePath()
		fill(dc, gg.Hex("#ff3040"))
		p.rect(dc, x-4, y-6, 8, 3, 1, gg.Hex("#40c040"))
	default:
		col := map[string]string{"orange": "#ff9a20", "apple": "#e02020", "melon": "#60d060", "galaxian": "#4080ff",
			"bell": "#ffd800", "key": "#a0e0ff"}[kind]
		if col == "" {
			col = "#ffffff"
		}
		dc.DrawCircle(x, y, 6)
		fill(dc, gg.Hex(col))
	}
}

// odds draws the last answer of a model-played Pac-Man at its junction: an
// arrow per option, the chosen one bright, with its probability -- jevman's
// "odds on the board".
func (p *Painter) odds(dc *gg.Context, s *Scene, ox, oy float64) {
	if s.Stats == nil || !s.Driven[game.Pacman] {
		return
	}
	d := s.Stats.Last[game.Pacman]
	if d.When.IsZero() || d.Escape || time.Since(d.When) > 3*time.Second {
		return
	}
	cx, cy := ox+px(float64(d.Key.X)), oy+px(float64(d.Key.Y))
	for dir := game.Up; dir <= game.Right; dir++ {
		pr := d.Probs[dir]
		if pr < 0 && dir != d.Choice {
			continue
		}
		dx, dy := dir.Vec()
		c := alpha(cAccent, 0.55)
		if dir == d.Choice {
			c = cAccent
		}
		ax, ay := cx+float64(dx)*Tile*1.1, cy+float64(dy)*Tile*1.1
		p.text(dc, arrows[dir], ax, ay, p.F.bold, 14, c, 0.5)
		if pr >= 0 {
			p.text(dc, fmt.Sprintf("%.0f%%", pr*100), ax+float64(dx)*Tile*0.9, ay+float64(dy)*Tile*0.7-boolF(dy == 0)*10,
				p.F.monoBold, 9, c, 0.5)
		}
	}
}

func boolF(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func (p *Painter) panel(dc *gg.Context, s *Scene) {
	x, y := float64(panelX), float64(panelY)
	p.rect(dc, x, y, panelW, panelH, 8, cPanel)
	x += 14
	p.text(dc, "AI decisions", x, y+22, p.F.bold, 16, cText, 0)
	explain := "The bars show the odds the model gave each way at its next junction; the brightest is the way it chose."
	p.text(dc, explain, x, y+42, p.F.sans, 11.5, cMuted, 0)
	cw := (panelW - 28 - 8) / 2.0
	cy := y + 58
	p.card(dc, s, game.Pacman, x, cy, panelW-28, 112)
	cy += 120
	ghostsAI := false
	for id := game.Blinky; id < game.NActors; id++ {
		ghostsAI = ghostsAI || s.Driven[id]
	}
	if ghostsAI {
		for i, id := range []int{game.Blinky, game.Pinky, game.Inky, game.Clyde} {
			p.card(dc, s, id, x+float64(i%2)*(cw+8), cy+float64(i/2)*120, cw, 112)
		}
		cy += 240
		p.totals(dc, s, x, cy)
		cy += 78
		p.text(dc, "Leaderboard", x, cy+8, p.F.bold, 13, cMuted, 0)
		p.miniBoard(dc, s, x, cy+20, panelW-28, panelY+panelH-cy-30)
		return
	}
	// The classic ghosts decide nothing: one line for the four, and the room
	// goes to the leaderboard, drawn as on its own tab.
	p.rect(dc, x, cy, panelW-28, 28, 6, cCard)
	gx := x + 10
	for id := game.Blinky; id < game.NActors; id++ {
		gx += p.text(dc, game.Names[id], gx, cy+14, p.F.bold, 12, Actors[id], 0) + 14
	}
	p.text(dc, "classic ghosts", x+panelW-36, cy+14, p.F.sans, 11, cMuted, 1)
	cy += 36
	p.totals(dc, s, x, cy)
	cy += 76
	p.text(dc, "Leaderboard", x, cy+8, p.F.bold, 14, cText, 0)
	p.text(dc, "mean score ± 2 SE · best · runs · mean latency", x+panelW-28, cy+8, p.F.sans, 11, cMuted, 1)
	cy += 22
	if len(s.Board) == 0 {
		p.text(dc, "No finished runs yet. Every game played here is kept.", x, cy+14, p.F.sans, 12, cMuted, 0)
		return
	}
	rows := min(len(s.Board), int((panelY+panelH-10-cy)/50))
	for i := 0; i < rows; i++ {
		p.barRow(dc, s.Board, i, x, cy+float64(i)*50, panelW-28, 44, s.Board[0].Mean, false)
	}
}

func (p *Painter) card(dc *gg.Context, s *Scene, id int, x, y, w, h float64) {
	col := Actors[id]
	p.rect(dc, x, y, w, h, 6, cCard)
	p.rect(dc, x, y, 3, h, 0, col)
	p.text(dc, game.Names[id], x+10, y+12, p.F.bold, 12, col, 0)
	var d play.Decision
	if s.Stats != nil {
		d = s.Stats.Last[id]
	}
	status, sc := "classic", cMuted
	switch {
	case id == game.Pacman && s.PacmanBy == "You":
		status = "you"
	case s.Thinking[id]:
		status, sc = "thinking…", cAccent
	case s.Driven[id] && d.When.IsZero():
		status = "AI · no decision yet"
	case s.Driven[id] && d.Fallback:
		status, sc = "greedy fallback", cDanger
	case s.Driven[id]:
		status = "AI"
		if d.Escape {
			status = "AI · escape"
		}
	}
	p.text(dc, status, x+w-8, y+12, p.F.sans, 11, sc, 1)
	fillC := col
	if d.Fallback {
		fillC = cDanger
	}
	for i, dir := range game.Dirs {
		by := y + 30 + float64(i)*16
		pr := d.Probs[dir]
		chosen := !d.When.IsZero() && d.Choice == dir
		ac := cMuted
		if chosen {
			ac = fillC
		}
		valid := pr >= 0 && !d.When.IsZero()
		if d.When.IsZero() {
			ac = alpha(cMuted, 0.4)
		}
		p.text(dc, arrows[dir], x+12, by+6, p.F.bold, 12, ac, 0.5)
		tx, tw := x+24, w-24-58
		p.rect(dc, tx, by+2, tw, 8, 3, cBorder)
		if valid {
			p.rect(dc, tx, by+2, math.Max(2, tw*pr), 8, 3, alpha(fillC, 0.35+0.65*boolF(chosen)))
		}
		label := "–"
		if valid {
			label = fmt.Sprintf("%.1f%%", pr*100)
		} else if chosen {
			label = "chosen"
		}
		p.text(dc, label, x+w-8, by+6, p.F.mono, 11, ac, 1)
	}
	meta := "no decision yet"
	if !d.When.IsZero() {
		if d.Fallback {
			meta = "the model did not answer in time"
		} else {
			meta = fmt.Sprintf("%s · %.0f ms", d.Choice, float64(d.Latency)/1e6)
			if d.Confidence > 0 {
				meta += fmt.Sprintf(" · confidence %.2f", d.Confidence)
			}
		}
	}
	p.text(dc, meta, x+10, y+h-10, p.F.sans, 10.5, cMuted, 0)
}

func (p *Painter) totals(dc *gg.Context, s *Scene, x, y float64) {
	type kv struct{ k, v string }
	st := s.Stats
	if st == nil {
		st = &play.Stats{}
	}
	items := []kv{
		{"decisions", strconv.Itoa(st.Decisions)},
		{"calls", strconv.Itoa(st.Calls)},
		{"mean latency", fmt.Sprintf("%.0f ms", float64(st.Mean())/1e6)},
		{"p50", fmt.Sprintf("%.0f ms", float64(st.P50())/1e6)},
		{"fallbacks", strconv.Itoa(st.Fallbacks)},
	}
	bw := (panelW - 28 - 4*6) / 5.0
	for i, it := range items {
		bx := x + float64(i)*(bw+6)
		p.rect(dc, bx, y, bw, 44, 6, cCard)
		p.text(dc, it.k, bx+8, y+13, p.F.sans, 10.5, cMuted, 0)
		c := cText
		if it.k == "fallbacks" && st.Fallbacks > 0 {
			c = cDanger
		}
		p.text(dc, it.v, bx+8, y+31, p.F.monoBold, 14, c, 0)
	}
	line := s.Summary
	if line == "" {
		line = "Pac-Man: " + s.PacmanBy + " · ghosts: " + s.GhostsBy
		if st.Err != nil {
			line = "model error: " + st.Err.Error()
		}
	}
	if len(line) > 90 {
		line = line[:89] + "…"
	}
	p.text(dc, line, x, y+60, p.F.sans, 11.5, cMuted, 0)
}

func (p *Painter) miniBoard(dc *gg.Context, s *Scene, x, y, w, h float64) {
	if len(s.Board) == 0 {
		p.text(dc, "No finished runs yet. Every game played here is kept.", x, y+14, p.F.sans, 12, cMuted, 0)
		return
	}
	rows := min(len(s.Board), int(h/26))
	top := s.Board[0].Mean
	for i := 0; i < rows; i++ {
		p.barRow(dc, s.Board, i, x, y+float64(i)*26, w, 22, top, true)
	}
}

func (p *Painter) barRow(dc *gg.Context, b []scores.Row, i int, x, y, w, h, top float64, small bool) {
	r := b[i]
	p.rect(dc, x, y, w, h, 8, cCard)
	frac := 0.0
	if top > 0 {
		frac = math.Max(0, math.Min(1, r.Mean/top))
	}
	if frac > 0 {
		gr := gg.NewLinearGradientBrush(x, y, x+w*frac, y).
			AddColorStop(0, gg.RGBA2(1, 0.85, 0, 0.12)).
			AddColorStop(1, gg.RGBA2(1, 0.85, 0, 0.45))
		dc.DrawRoundedRectangle(x, y, w*frac, h, 8)
		dc.SetFillBrush(gr)
		dc.Fill()
	}
	rank := 1
	for j := 0; j < i; j++ {
		if b[j].Mean-2*b[j].SE > r.Mean+2*r.SE {
			rank++
		}
	}
	size, big := 12.0, 13.0
	if !small {
		size, big = 15, 19
	}
	p.text(dc, strconv.Itoa(rank), x+14, y+h/2, p.F.monoBold, big, cAccent, 0)
	p.text(dc, r.Label(), x+40, y+h/2, p.F.bold, big, cText, 0)
	right := fmt.Sprintf("%.0f", r.Mean)
	p.text(dc, right, x+w-14, y+h/2-boolF(!small)*8, p.F.monoBold, big, cText, 1)
	detail := fmt.Sprintf("± %.0f · best %d · %d run", 2*r.SE, r.Best, r.Runs)
	if r.Runs != 1 {
		detail += "s"
	}
	if r.AvgMs > 0 {
		detail += fmt.Sprintf(" · %.0f ms", r.AvgMs)
	}
	if small {
		p.text(dc, detail, x+w-80, y+h/2, p.F.sans, size-1, cMuted, 1)
	} else {
		p.text(dc, detail, x+w-14, y+h/2+12, p.F.sans, 11.5, cMuted, 1)
	}
}

func (p *Painter) leaderboard(dc *gg.Context, s *Scene) {
	x, y := float64(panelX), float64(panelY)
	p.rect(dc, x, y, panelW, panelH, 8, cPanel)
	x += 14
	p.text(dc, "Which model plays Pac-Man best?", x, y+26, p.F.bold, 20, cText, 0)
	p.text(dc, "Every finished game on this machine, against the classic ghosts. Mean score ± two standard errors;",
		x, y+52, p.F.sans, 12, cMuted, 0)
	p.text(dc, "a rank is shared until the intervals part.", x, y+68, p.F.sans, 12, cMuted, 0)
	if len(s.Board) == 0 {
		p.text(dc, "No finished runs yet.", x, y+110, p.F.sans, 14, cMuted, 0)
		return
	}
	top := s.Board[0].Mean
	for i := 0; i < min(len(s.Board), 11); i++ {
		p.barRow(dc, s.Board, i, x, y+90+float64(i)*54, panelW-28, 46, top, false)
	}
}

func (p *Painter) dialog(dc *gg.Context, s *Scene) {
	d := s.Dialog
	p.rect(dc, 0, 0, W, H, 0, gg.RGBA2(0, 0, 0, 0.6))
	s.hit(0, 0, W, H, Action{Kind: ActNone})
	cw := 560.0
	// Each side's chips wrap at about four a row.
	ch := 290 + float64(len(d.Models)/4)*80
	if d.Status != "" {
		ch += 24
	}
	x, y := (W-cw)/2, (H-ch)/2
	p.rect(dc, x, y, cw, ch, 10, cPanel)
	dc.DrawRoundedRectangle(x+0.5, y+0.5, cw-1, ch-1, 10)
	dc.SetRGBA(0.165, 0.165, 0.25, 1)
	dc.SetLineWidth(1)
	dc.Stroke()
	p.text(dc, "×", x+cw-20, y+20, p.F.sans, 20, cMuted, 0.5)
	s.hit(x+cw-36, y+4, 32, 32, Action{Kind: ActCloseDialog})
	p.text(dc, "Play", x+cw/2, y+30, p.F.bold, 20, cText, 0.5)
	p.text(dc, "Pick who plays each side. The models are the decision models in your models folder.",
		x+cw/2, y+54, p.F.sans, 12, cMuted, 0.5)
	cy := y + 80
	chips := func(title string, first string, sel int, kind ActionKind) {
		p.text(dc, title, x+24, cy+8, p.F.bold, 13, cMuted, 0)
		cy += 22
		cx := x + 24
		items := append([]string{first}, d.Models...)
		for i, name := range items {
			w := p.measure(dc, name, p.F.sans, 13) + 24
			if cx+w > x+cw-24 {
				cx = x + 24
				cy += 40
			}
			on := i-1 == sel
			bg, fg := cButton, cText
			if on {
				bg, fg = cAccent, cBlack
			}
			p.rect(dc, cx, cy, w, 32, 16, bg)
			p.text(dc, name, cx+12, cy+16, p.F.sans, 13, fg, 0)
			s.hit(cx, cy, w, 32, Action{kind, i - 1})
			cx += w + 8
		}
		cy += 50
	}
	chips("Pac-Man", "You (arrow keys)", d.Pacman, ActPickPacman)
	chips("Ghosts", "Classic", d.Ghosts, ActPickGhosts)
	label := "Play"
	if d.Busy {
		label = "Loading…"
	}
	bw := 160.0
	p.rect(dc, x+(cw-bw)/2, cy, bw, 40, 8, cAccent)
	p.text(dc, label, x+cw/2, cy+20, p.F.bold, 16, cBlack, 0.5)
	if !d.Busy {
		s.hit(x+(cw-bw)/2, cy, bw, 40, Action{Kind: ActStart})
	}
	if d.Status != "" {
		c := cMuted
		if len(d.Status) > 6 && d.Status[:6] == "error:" {
			c = cDanger
		}
		st := d.Status
		if len(st) > 100 {
			st = st[:99] + "…"
		}
		p.text(dc, st, x+cw/2, cy+58, p.F.sans, 12, c, 0.5)
	}
}
