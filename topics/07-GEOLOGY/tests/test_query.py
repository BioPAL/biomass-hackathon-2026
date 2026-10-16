import pytest

from biomass import QueryError, SearchQuery


def test_cql_filter_combines_product_type_and_extra_filters():
    q = SearchQuery("BiomassLevel1a", "S1_SCS__1S",
                    extra_filters={"sat:orbit_state": "DESCENDING", "eofeos:orbit_drift_flag": False})
    assert q.cql_filter() == ("product:type='S1_SCS__1S' AND sat:orbit_state='DESCENDING' "
                              "AND eofeos:orbit_drift_flag=false")


def test_cql_filter_is_none_without_filters():
    assert SearchQuery("BiomassLevel1a").cql_filter() is None


def test_datetime_range():
    q = SearchQuery("c", date_start="2026-01-01", date_end="2026-07-01")
    assert q.datetime_range() == ["2026-01-01T00:00:00Z", "2026-07-01T23:59:59Z"]


def test_resolved_bbox_returns_list():
    q = SearchQuery("c", bbox=(-81, -20, -35, 10))
    assert q.resolved_bbox() == [-81, -20, -35, 10]


@pytest.mark.parametrize("kwargs", [
    {"date_start": "2026-07-01", "date_end": "2026-01-01"},
    {"date_start": "01/01/2026"},
    {"bbox": [-81, -20, -35]},
    {"bbox": [-35, -20, -81, 10]},
    {"max_items": 0},
])
def test_invalid_queries_raise(kwargs):
    with pytest.raises(QueryError):
        SearchQuery("c", **kwargs)
