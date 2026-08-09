import duckdb

from rigged_matchup_ml.prepare import _ranked_segment_sql


def test_ranked_segment_sql_pools_prepared_rows_only() -> None:
    expression = _ranked_segment_sql({"ranked_league_buckets": [1, 3, 5, 8]})
    rows = duckdb.sql(
        f"""
        select {expression} as mapped
        from (values
          ('ranked', 'ranked:league-1'),
          ('ranked', 'ranked:league-4'),
          ('ranked', 'ranked:league-7'),
          ('ranked', 'ranked:unknown'),
          ('ladder', 'ladder:5000-6999')
        ) source(mode_key, segment)
        """
    ).fetchall()
    assert [row[0] for row in rows] == [
        "ranked:league-1-2",
        "ranked:league-3-4",
        "ranked:league-5-7",
        "ranked:unknown",
        "ladder:5000-6999",
    ]
