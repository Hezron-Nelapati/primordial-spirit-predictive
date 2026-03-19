use rusqlite::Connection;
use crate::graph::{GraphAccess, NodeId, WordEdge, WordNode};

// ---------------------------------------------------------------------------
// GraphDb — SQLite-backed graph store
//
// Schema
// ------
// nodes(id, surface, frequency, pos_x, pos_y, pos_z, lv0..lv4)
// edges(id, from_id, to_id, weight, intent, tone, domain, entity, dated)
//
// `dated` is stored as INTEGER with -1 as the sentinel for "no date" (NULL
// would prevent deduplication via UNIQUE INDEX since NULL != NULL in SQL).
//
// Edge deduplication mirrors ingest_v2_rows: same (from, to, intent, domain,
// dated) tuple → reinforce weight by 1.0 rather than inserting a duplicate.
// ---------------------------------------------------------------------------

pub struct GraphDb {
    pub(crate) conn: Connection,
}

impl GraphDb {
    /// Open (or create) a persistent database file.
    pub fn open(path: &str) -> rusqlite::Result<Self> {
        let conn = Connection::open(path)?;
        // busy_timeout: SQLite will retry internally for up to 5 s if another
        // process holds a write lock, then return SQLITE_BUSY as an Err.
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; PRAGMA busy_timeout=5000;",
        )?;
        let db = Self { conn };
        db.create_schema()?;
        Ok(db)
    }

    /// Open a transient in-memory database (useful for tests).
    pub fn open_memory() -> rusqlite::Result<Self> {
        let db = Self { conn: Connection::open_in_memory()? };
        db.create_schema()?;
        Ok(db)
    }

    fn create_schema(&self) -> rusqlite::Result<()> {
        self.conn.execute_batch("
            CREATE TABLE IF NOT EXISTS nodes (
                id        INTEGER PRIMARY KEY,
                surface   TEXT    NOT NULL UNIQUE,
                frequency INTEGER NOT NULL DEFAULT 1,
                pos_x     REAL    NOT NULL DEFAULT 0,
                pos_y     REAL    NOT NULL DEFAULT 0,
                pos_z     REAL    NOT NULL DEFAULT 0,
                lv0 REAL NOT NULL DEFAULT 0,
                lv1 REAL NOT NULL DEFAULT 0,
                lv2 REAL NOT NULL DEFAULT 0,
                lv3 REAL NOT NULL DEFAULT 0,
                lv4 REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS edges (
                id      INTEGER PRIMARY KEY,
                from_id INTEGER NOT NULL,
                to_id   INTEGER NOT NULL,
                weight  REAL    NOT NULL DEFAULT 1.0,
                intent  TEXT    NOT NULL DEFAULT '',
                tone    TEXT    NOT NULL DEFAULT '',
                domain  TEXT    NOT NULL DEFAULT '',
                entity  TEXT,
                dated   INTEGER NOT NULL DEFAULT -1
            );
            CREATE INDEX IF NOT EXISTS idx_edges_from ON edges(from_id);
            CREATE INDEX IF NOT EXISTS idx_edges_to   ON edges(to_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_edge_dedup
                ON edges(from_id, to_id, intent, domain, dated);
        ")
    }

    /// Insert or increment frequency for a node.
    pub fn upsert_node(&self, node: &WordNode) -> rusqlite::Result<()> {
        self.conn.execute(
            "INSERT INTO nodes (id, surface, frequency, pos_x, pos_y, pos_z,
                                lv0, lv1, lv2, lv3, lv4)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)
             ON CONFLICT(id) DO UPDATE SET frequency = frequency + excluded.frequency",
            rusqlite::params![
                node.id as i64, node.surface, node.frequency as i64,
                node.position[0], node.position[1], node.position[2],
                node.lexical_vector[0], node.lexical_vector[1],
                node.lexical_vector[2], node.lexical_vector[3],
                node.lexical_vector[4],
            ],
        )?;
        Ok(())
    }

    /// Insert edge or reinforce weight if the dedup key already exists.
    pub fn upsert_edge(
        &self,
        from: NodeId, to: NodeId,
        weight: f32,
        intent: &str, tone: &str, domain: &str,
        entity: Option<&str>,
        dated: Option<u16>,
    ) -> rusqlite::Result<()> {
        let dated_i64 = dated.map(|y| y as i64).unwrap_or(-1);
        self.conn.execute(
            "INSERT INTO edges (from_id, to_id, weight, intent, tone, domain, entity, dated)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)
             ON CONFLICT(from_id, to_id, intent, domain, dated)
             DO UPDATE SET weight = weight + excluded.weight",
            rusqlite::params![from as i64, to as i64, weight,
                              intent, tone, domain, entity, dated_i64],
        )?;
        Ok(())
    }

    pub fn begin(&self) -> rusqlite::Result<()> {
        self.conn.execute_batch("BEGIN")
    }
    pub fn commit(&self) -> rusqlite::Result<()> {
        self.conn.execute_batch("COMMIT")
    }

    fn row_to_node(row: &rusqlite::Row<'_>) -> rusqlite::Result<WordNode> {
        Ok(WordNode {
            id:       row.get::<_, i64>(0)? as NodeId,
            surface:  row.get(1)?,
            frequency: row.get::<_, i64>(2)? as u32,
            position: [row.get(3)?, row.get(4)?, row.get(5)?],
            lexical_vector: [row.get(6)?, row.get(7)?, row.get(8)?, row.get(9)?, row.get(10)?],
        })
    }

    fn row_to_edge(row: &rusqlite::Row<'_>) -> rusqlite::Result<WordEdge> {
        // Columns: from_id(0), to_id(1), weight(2), intent(3), tone(4), domain(5), entity(6), dated(7)
        let dated_i64: i64 = row.get(7)?;
        Ok(WordEdge {
            from:   row.get::<_, i64>(0)? as NodeId,
            to:     row.get::<_, i64>(1)? as NodeId,
            weight: row.get(2)?,
            intent: row.get(3)?,
            tone:   row.get(4)?,
            domain: row.get(5)?,
            entity: row.get(6)?,
            dated:  if dated_i64 < 0 { None } else { Some(dated_i64 as u16) },
        })
    }
}

impl GraphAccess for GraphDb {
    fn surface_to_id(&self, surface: &str) -> Option<NodeId> {
        self.conn.query_row(
            "SELECT id FROM nodes WHERE surface = ?1",
            [surface],
            |row| row.get::<_, i64>(0).map(|id| id as NodeId),
        ).ok()
    }

    fn node_by_id(&self, id: NodeId) -> Option<WordNode> {
        self.conn.query_row(
            "SELECT id, surface, frequency, pos_x, pos_y, pos_z,
                    lv0, lv1, lv2, lv3, lv4
             FROM nodes WHERE id = ?1",
            [id as i64],
            Self::row_to_node,
        ).ok()
    }

    fn edges_from(&self, from_id: NodeId) -> Vec<WordEdge> {
        let mut stmt = match self.conn.prepare_cached(
            "SELECT from_id, to_id, weight, intent, tone, domain, entity, dated
             FROM edges WHERE from_id = ?1",
        ) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        stmt.query_map([from_id as i64], Self::row_to_edge)
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    fn edges_to(&self, to_id: NodeId) -> Vec<WordEdge> {
        let mut stmt = match self.conn.prepare_cached(
            "SELECT from_id, to_id, weight, intent, tone, domain, entity, dated
             FROM edges WHERE to_id = ?1",
        ) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        stmt.query_map([to_id as i64], Self::row_to_edge)
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    fn has_edges_from(&self, id: NodeId) -> bool {
        self.conn.query_row(
            "SELECT 1 FROM edges WHERE from_id = ?1 LIMIT 1",
            [id as i64],
            |_| Ok(()),
        ).is_ok()
    }

    fn all_nodes(&self) -> Vec<WordNode> {
        let mut stmt = match self.conn.prepare(
            "SELECT id, surface, frequency, pos_x, pos_y, pos_z,
                    lv0, lv1, lv2, lv3, lv4 FROM nodes",
        ) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        stmt.query_map([], Self::row_to_node)
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
            .unwrap_or_default()
    }

    fn node_count(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM nodes", [], |r| r.get::<_, i64>(0))
            .map(|n| n as usize)
            .unwrap_or(0)
    }

    fn edge_count(&self) -> usize {
        self.conn
            .query_row("SELECT COUNT(*) FROM edges", [], |r| r.get::<_, i64>(0))
            .map(|n| n as usize)
            .unwrap_or(0)
    }
}
