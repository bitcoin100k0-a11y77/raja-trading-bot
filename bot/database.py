"""
Raja Banks Trading Agent - Database Layer
SQLite with WAL mode for concurrent reads. Persistent across restarts.
"""
import sqlite3
import os
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from constants import EntryType, TradeDirection, ExitReason, LossCategory
from utils import utc_now

logger = logging.getLogger(__name__)


class TradingDatabase:
    """SQLite database for trade journal, loss analysis, and adaptive learning."""

    def __init__(self, db_path: str = "data/trading_bot.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        try:
            self.conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        self._create_tables()
        logger.info("Database initialized: %s", db_path)

    def _create_tables(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE NOT NULL,
            symbol TEXT NOT NULL DEFAULT 'XAUUSD',
            direction TEXT NOT NULL,
            entry_type TEXT NOT NULL,
            entry_price REAL NOT NULL,
            lot_size REAL NOT NULL,
            sl_initial REAL NOT NULL,
            sl_current REAL NOT NULL,
            sl_stage INTEGER DEFAULT 1,
            tp1_price REAL,
            tp2_price REAL,
            tp1_hit INTEGER DEFAULT 0,
            exit_price REAL,
            exit_reason TEXT,
            exit_time TEXT,
            risk_pips REAL,
            pnl_pips REAL DEFAULT 0.0,
            pnl_dollars REAL DEFAULT 0.0,
            rr_achieved REAL DEFAULT 0.0,
            risk_percent REAL,
            balance_before REAL,
            balance_after REAL,
            entry_time TEXT NOT NULL,
            session_type TEXT,
            four_h_candle INTEGER,
            is_open INTEGER DEFAULT 1,
            partial_close_pct REAL DEFAULT 0.0,
            created_at TEXT DEFAULT (datetime('now'))
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_context (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT NOT NULL UNIQUE,
            sr_level_price REAL,
            sr_level_type TEXT,
            sr_touches INTEGER,
            distance_to_sr_pips REAL,
            c1_body_ratio REAL,
            c1_range_pips REAL,
            c1_wick_type TEXT,
            c1_is_wickless INTEGER DEFAULT 0,
            trend_type TEXT,
            htf_zone_aligned INTEGER DEFAULT 0,
            room_to_opposing_pips REAL,
            mid_range_position REAL,
            volume_strength TEXT,
            spread_pips REAL,
            confidence_score REAL DEFAULT 1.0,
            blocked_patterns_matched INTEGER DEFAULT 0,
            impulse_body_ratio REAL,
            impulse_volume_ratio REAL,
            clean_range_bars INTEGER,
            break_level_price REAL,
            retest_bars_waited INTEGER,
            FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS loss_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT NOT NULL,
            category TEXT NOT NULL,
            notes TEXT,
            assigned_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_key TEXT UNIQUE NOT NULL,
            description TEXT,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            losing_trades INTEGER DEFAULT 0,
            loss_rate REAL DEFAULT 0.0,
            is_blocked INTEGER DEFAULT 0,
            blocked_at TEXT,
            last_updated TEXT DEFAULT (datetime('now'))
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            trades_taken INTEGER DEFAULT 0,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            pnl_dollars REAL DEFAULT 0.0,
            max_drawdown_pct REAL DEFAULT 0.0,
            balance_start REAL,
            balance_end REAL,
            risk_level REAL DEFAULT 1.0,
            consecutive_losses INTEGER DEFAULT 0
        )
        """)

        self.conn.commit()

    # === TRADE CRUD ===

    def insert_trade(self, trade: Dict[str, Any]) -> str:
        cols = ", ".join(trade.keys())
        placeholders = ", ".join(["?"] * len(trade))
        self.conn.execute(
            f"INSERT INTO trades ({cols}) VALUES ({placeholders})",
            list(trade.values())
        )
        self.conn.commit()
        return trade["trade_id"]

    def update_trade(self, trade_id: str, updates: Dict[str, Any]):
        set_clause = ", ".join(f"{k} = ?" for k in updates.keys())
        self.conn.execute(
            f"UPDATE trades SET {set_clause} WHERE trade_id = ?",
            list(updates.values()) + [trade_id]
        )
        self.conn.commit()

    def close_trade(self, trade_id: str, exit_price: float,
                    exit_reason: str, pnl_pips: float, pnl_dollars: float,
                    rr_achieved: float, balance_after: float):
        self.update_trade(trade_id, {
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "exit_time": utc_now().isoformat(),
            "pnl_pips": pnl_pips,
            "pnl_dollars": pnl_dollars,
            "rr_achieved": rr_achieved,
            "balance_after": balance_after,
            "is_open": 0,
        })

    def get_open_trades(self) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM trades WHERE is_open = 1").fetchall()
        return [dict(r) for r in rows]

    def get_trade(self, trade_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM trades ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # === TRADE CONTEXT ===

    def insert_context(self, context: Dict[str, Any]):
        cols = ", ".join(context.keys())
        placeholders = ", ".join(["?"] * len(context))
        self.conn.execute(
            f"INSERT INTO trade_context ({cols}) VALUES ({placeholders})",
            list(context.values())
        )
        self.conn.commit()

    def get_context(self, trade_id: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM trade_context WHERE trade_id = ?", (trade_id,)
        ).fetchone()
        return dict(row) if row else None

    # === LOSS CATEGORIES ===

    def assign_loss_category(self, trade_id: str, category: str, notes: str = ""):
        self.conn.execute(
            "INSERT INTO loss_categories (trade_id, category, notes) VALUES (?, ?, ?)",
            (trade_id, category, notes)
        )
        self.conn.commit()

    def get_loss_categories(self, trade_id: str) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM loss_categories WHERE trade_id = ?", (trade_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_category_stats(self) -> List[Dict]:
        rows = self.conn.execute("""
            SELECT category, COUNT(*) as count,
                   GROUP_CONCAT(trade_id) as trade_ids
            FROM loss_categories
            GROUP BY category
            ORDER BY count DESC
        """).fetchall()
        return [dict(r) for r in rows]

    # === PATTERN LEARNING ===

    def upsert_pattern(self, pattern_key: str, is_win: bool, description: str = ""):
        existing = self.conn.execute(
            "SELECT * FROM patterns WHERE pattern_key = ?", (pattern_key,)
        ).fetchone()
        now = utc_now().isoformat()
        if existing:
            total = existing["total_trades"] + 1
            wins = existing["winning_trades"] + (1 if is_win else 0)
            losses = existing["losing_trades"] + (0 if is_win else 1)
            loss_rate = losses / total if total > 0 else 0.0
            self.conn.execute("""
                UPDATE patterns SET total_trades = ?, winning_trades = ?,
                    losing_trades = ?, loss_rate = ?, last_updated = ?
                WHERE pattern_key = ?
            """, (total, wins, losses, loss_rate, now, pattern_key))
        else:
            wins = 1 if is_win else 0
            losses = 0 if is_win else 1
            loss_rate = losses / 1
            self.conn.execute("""
                INSERT INTO patterns
                    (pattern_key, description, total_trades, winning_trades,
                     losing_trades, loss_rate, last_updated)
                VALUES (?, ?, 1, ?, ?, ?, ?)
            """, (pattern_key, description, wins, losses, loss_rate, now))
        self.conn.commit()

    def block_pattern(self, pattern_key: str):
        self.conn.execute("""
            UPDATE patterns SET is_blocked = 1, blocked_at = ?
            WHERE pattern_key = ?
        """, (utc_now().isoformat(), pattern_key))
        self.conn.commit()

    def unblock_pattern(self, pattern_key: str):
        self.conn.execute(
            "UPDATE patterns SET is_blocked = 0, blocked_at = NULL WHERE pattern_key = ?",
            (pattern_key,)
        )
        self.conn.commit()

    def get_blocked_patterns(self) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM patterns WHERE is_blocked = 1").fetchall()
        return [dict(r) for r in rows]

    def get_all_patterns(self) -> List[Dict]:
        rows = self.conn.execute("SELECT * FROM patterns ORDER BY loss_rate DESC").fetchall()
        return [dict(r) for r in rows]

    # === DAILY STATS ===

    def upsert_daily_stats(self, date_str: str, stats: Dict[str, Any]):
        existing = self.conn.execute(
            "SELECT * FROM daily_stats WHERE date = ?", (date_str,)
        ).fetchone()
        if existing:
            set_clause = ", ".join(f"{k} = ?" for k in stats.keys())
            self.conn.execute(
                f"UPDATE daily_stats SET {set_clause} WHERE date = ?",
                list(stats.values()) + [date_str]
            )
        else:
            stats["date"] = date_str
            cols = ", ".join(stats.keys())
            placeholders = ", ".join(["?"] * len(stats))
            self.conn.execute(
                f"INSERT INTO daily_stats ({cols}) VALUES ({placeholders})",
                list(stats.values())
            )
        self.conn.commit()

    def get_daily_stats(self, date_str: str) -> Optional[Dict]:
        row = self.conn.execute(
            "SELECT * FROM daily_stats WHERE date = ?", (date_str,)
        ).fetchone()
        return dict(row) if row else None

    # === ANALYTICS ===

    def get_consecutive_losses(self) -> int:
        rows = self.conn.execute("""
            SELECT pnl_dollars FROM trades
            WHERE is_open = 0 ORDER BY exit_time DESC LIMIT 20
        """).fetchall()
        count = 0
        for r in rows:
            if r["pnl_dollars"] < 0:
                count += 1
            else:
                break
        return count

    def get_win_rate(self, last_n: int = 50) -> float:
        rows = self.conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as wins
            FROM (SELECT pnl_dollars FROM trades WHERE is_open = 0
                  ORDER BY exit_time DESC LIMIT ?)
        """, (last_n,)).fetchone()
        if rows["total"] == 0:
            return 0.0
        return rows["wins"] / rows["total"]

    def get_total_trades(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM trades").fetchone()
        return row["cnt"]

    def get_trades_today(self) -> int:
        today = utc_now().strftime("%Y-%m-%d")
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM trades WHERE entry_time LIKE ?",
            (f"{today}%",)
        ).fetchone()
        return row["cnt"]

    def get_daily_pnl(self) -> float:
        today = utc_now().strftime("%Y-%m-%d")
        row = self.conn.execute(
            "SELECT COALESCE(SUM(pnl_dollars), 0) as total FROM trades "
            "WHERE exit_time LIKE ? AND is_open = 0",
            (f"{today}%",)
        ).fetchone()
        return row["total"]

    def get_entry_type_stats(self) -> List[Dict]:
        rows = self.conn.execute("""
            SELECT entry_type, COUNT(*) as total,
                   SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl_dollars <= 0 THEN 1 ELSE 0 END) as losses,
                   AVG(rr_achieved) as avg_rr,
                   SUM(pnl_dollars) as total_pnl
            FROM trades WHERE is_open = 0 GROUP BY entry_type
        """).fetchall()
        return [dict(r) for r in rows]

    def get_session_stats(self) -> List[Dict]:
        rows = self.conn.execute("""
            SELECT session_type, COUNT(*) as total,
                   SUM(CASE WHEN pnl_dollars > 0 THEN 1 ELSE 0 END) as wins,
                   SUM(pnl_dollars) as total_pnl
            FROM trades WHERE is_open = 0 GROUP BY session_type
        """).fetchall()
        return [dict(r) for r in rows]

    def get_learning_summary(self) -> Dict:
        """Get summary for Telegram daily report."""
        patterns = self.get_all_patterns()
        blocked = [p for p in patterns if p["is_blocked"]]
        total_trades = self.get_total_trades()
        win_rate = self.get_win_rate(50)

        return {
            "total_trades": total_trades,
            "win_rate_50": round(win_rate * 100, 1),
            "patterns_tracked": len(patterns),
            "patterns_blocked": len(blocked),
            "blocked_details": [
                {"key": p["pattern_key"], "loss_rate": round(p["loss_rate"] * 100),
                 "trades": p["total_trades"]}
                for p in blocked
            ],
        }

    def close(self):
        self.conn.close()
