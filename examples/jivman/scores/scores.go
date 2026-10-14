// Package scores is jivman's leaderboard: every finished run in a musql
// database (github.com/samyfodil/musql, embedded through database/sql), and
// the board read back from it -- per player, the runs, the best and the mean
// score with two standard errors, and the mean decision latency.
package scores

import (
	"database/sql"
	"math"
	"os"
	"path/filepath"
	"time"

	_ "github.com/samyfodil/musql/driver"
)

// The modes a run is played in.
const (
	// ModeAI is a model playing Pac-Man against the classic ghosts.
	ModeAI = "ai"
	// ModeHuman is a person playing Pac-Man against the classic ghosts: the
	// same game, so it ranks on the same board.
	ModeHuman = "human"
	// ModeGhosts is a person playing Pac-Man against a model's ghosts; the
	// score is the person's.
	ModeGhosts = "ghosts"
)

// Run is one finished game.
type Run struct {
	Mode string
	// Who is the model's name, or "human".
	Who       string
	Score     int
	Level     int
	Ticks     int
	Decisions int
	Fallbacks int
	// AvgMs and P50Ms are the model's decision latency (0 for a human).
	AvgMs  float64
	P50Ms  float64
	Device string
	At     time.Time
}

// Row is one player's line on the board.
type Row struct {
	Mode string
	Who  string
	Runs int
	Best int
	Mean float64
	// SE is the standard error of the mean score (0 for a single run).
	SE    float64
	AvgMs float64
}

// Store is the database.
type Store struct{ db *sql.DB }

// DefaultPath is where the scores live: jitllm's folder in this system's
// config directory.
func DefaultPath() string {
	dir, err := os.UserConfigDir()
	if err != nil {
		dir = "."
	}
	return filepath.Join(dir, "jitllm", "jivman", "scores.musq")
}

const schema = `CREATE TABLE IF NOT EXISTS runs (
	id INTEGER PRIMARY KEY,
	mode TEXT NOT NULL,
	who TEXT NOT NULL,
	score INTEGER NOT NULL,
	level INTEGER NOT NULL,
	ticks INTEGER NOT NULL,
	decisions INTEGER NOT NULL,
	fallbacks INTEGER NOT NULL,
	avg_ms REAL NOT NULL,
	p50_ms REAL NOT NULL,
	device TEXT NOT NULL,
	at INTEGER NOT NULL
)`

// Open opens (creating if missing) the database at path.
func Open(path string) (*Store, error) {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return nil, err
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, err
	}
	return &Store{db: db}, nil
}

// Close closes the database.
func (s *Store) Close() error { return s.db.Close() }

// Add records a run.
func (s *Store) Add(r Run) error {
	if r.At.IsZero() {
		r.At = time.Now()
	}
	_, err := s.db.Exec(`INSERT INTO runs (mode, who, score, level, ticks, decisions, fallbacks, avg_ms, p50_ms, device, at)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		r.Mode, r.Who, r.Score, r.Level, r.Ticks, r.Decisions, r.Fallbacks, r.AvgMs, r.P50Ms, r.Device, r.At.Unix())
	return err
}

// Board is every player's line, best mean first. The mean, the mean of the
// squares and the count come from the database; the standard error is
// worked out from them here.
func (s *Store) Board() ([]Row, error) {
	rows, err := s.db.Query(`SELECT mode, who, COUNT(*), MAX(score), AVG(score), AVG(score * score), AVG(avg_ms)
		FROM runs GROUP BY mode, who ORDER BY AVG(score) DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []Row
	for rows.Next() {
		var r Row
		var sq float64
		if err := rows.Scan(&r.Mode, &r.Who, &r.Runs, &r.Best, &r.Mean, &sq, &r.AvgMs); err != nil {
			return nil, err
		}
		if r.Runs > 1 {
			n := float64(r.Runs)
			variance := max(0, sq-r.Mean*r.Mean) * n / (n - 1)
			r.SE = math.Sqrt(variance / n)
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

// HighScore is the best score of any run.
func (s *Store) HighScore() (int, error) {
	var hs sql.NullInt64
	err := s.db.QueryRow(`SELECT MAX(score) FROM runs`).Scan(&hs)
	return int(hs.Int64), err
}

// Label is how a row reads on the board.
func (r Row) Label() string {
	switch r.Mode {
	case ModeHuman:
		return "You"
	case ModeGhosts:
		return "You vs " + r.Who + " ghosts"
	}
	return r.Who
}
