-- World Cup 2026 prediction tool — SQLite schema
-- One file = the full database shape. Safe to re-run (CREATE IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS teams (
    team_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT UNIQUE NOT NULL,
    fifa_code     TEXT,
    confederation TEXT
);

CREATE TABLE IF NOT EXISTS matches (
    match_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    date        TEXT NOT NULL,          -- ISO 'YYYY-MM-DD'
    competition TEXT,                   -- 'Friendly','FIFA World Cup qualification','FIFA World Cup',...
    home_id     INTEGER NOT NULL,
    away_id     INTEGER NOT NULL,
    home_goals  INTEGER,                -- NULL until played
    away_goals  INTEGER,
    neutral     INTEGER DEFAULT 0,      -- 0/1 boolean
    stage       TEXT,                   -- 'group','R32','R16','QF','SF','F', NULL
    status      TEXT DEFAULT 'finished',-- 'scheduled','live','finished'
    source      TEXT,                   -- where the row came from: 'history_csv','football-data',...
    FOREIGN KEY (home_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_id) REFERENCES teams(team_id),
    UNIQUE(date, home_id, away_id)
);

CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_home ON matches(home_id);
CREATE INDEX IF NOT EXISTS idx_matches_away ON matches(away_id);

CREATE TABLE IF NOT EXISTS elo_ratings (
    team_id     INTEGER NOT NULL,
    as_of_date  TEXT NOT NULL,
    rating      REAL NOT NULL,
    PRIMARY KEY (team_id, as_of_date),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS team_strengths (
    team_id     INTEGER NOT NULL,
    as_of_date  TEXT NOT NULL,
    attack      REAL,
    defense     REAL,
    PRIMARY KEY (team_id, as_of_date),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE IF NOT EXISTS predictions (
    match_id        INTEGER PRIMARY KEY,
    p_home          REAL,
    p_draw          REAL,
    p_away          REAL,
    exp_home_goals  REAL,
    exp_away_goals  REAL,
    top_score       TEXT,
    p_btts          REAL,
    p_over25        REAL,
    model_version   TEXT,
    created_at      TEXT,
    FOREIGN KEY (match_id) REFERENCES matches(match_id)
);
