"""Film metadata from Wikidata (and optionally TMDb), written as a Kodi-style .nfo.

The .nfo carries the film's IMDb, TMDb and Wikidata ids, so Kodi, Jellyfin,
Plex and tesserae can fetch anything it leaves out (a "combination NFO").

Posters: Wikidata only holds freely licensed ones, which few in-copyright
films have. Set TMDB_API_KEY (a free v3 key from themoviedb.org) to fetch
the poster and fan art from TMDb instead.
"""

import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

USER_AGENT = "media-ripper/0.1 (https://github.com/misterbisson/media-ripper)"
SPARQL = "https://query.wikidata.org/sparql"
FILM = "wd:Q11424"
# Wikidata duration units, in minutes
UNIT_MINUTES = {"Q7727": 1, "Q11574": 1 / 60, "Q25235": 60}


@dataclass
class Film:
    qid: str
    title: str
    description: str = ""
    year: int | None = None
    premiered: str | None = None  # only when Wikidata knows the day
    minutes: float | None = None
    original_title: str | None = None
    directors: list[str] = field(default_factory=list)
    writers: list[str] = field(default_factory=list)
    cast: list[str] = field(default_factory=list)
    genres: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    studios: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)  # ISO 639-2 codes
    imdb: str | None = None
    tmdb: str | None = None
    poster_url: str | None = None
    enwiki: str | None = None
    plot: str | None = None

    @property
    def folder_name(self):
        return f"{self.title} ({self.year})" if self.year else self.title

    def summary(self):
        bits = [str(self.year) if self.year else "",
                ", ".join(self.directors[:2]),
                f"{round(self.minutes)} min" if self.minutes else "",
                self.description]
        return f"{self.title} — " + " · ".join(b for b in bits if b)


# ---------------------------------------------------------------- HTTP

def get(url, params=None, headers=None, raw=False):
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = resp.read()
        return (body, resp.headers.get_content_type()) if raw else json.loads(body)


def sparql(query):
    data = get(SPARQL, {"query": query}, {"Accept": "application/sparql-results+json"})
    return [{k: v["value"] for k, v in row.items()}
            for row in data["results"]["bindings"]]


def qid(uri):
    return uri.rsplit("/", 1)[-1]


def literal(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ---------------------------------------------------------------- search

def search(name, runtime_secs=None):
    """Films whose name matches, best first: close runtime, then search rank."""
    rows = sparql(f"""
        SELECT ?item ?ordinal ?label ?desc ?date ?amount ?unit ?dirLabel WHERE {{
          SERVICE wikibase:mwapi {{
            bd:serviceParam wikibase:endpoint "www.wikidata.org";
                            wikibase:api "EntitySearch";
                            mwapi:search {literal(name)}; mwapi:language "en";
                            mwapi:limit "20".
            ?item wikibase:apiOutputItem mwapi:item.
            ?ordinal wikibase:apiOrdinal true.
          }}
          ?item wdt:P31/wdt:P279* {FILM}.
          OPTIONAL {{ ?item rdfs:label ?label FILTER(LANG(?label) = "en") }}
          OPTIONAL {{ ?item schema:description ?desc FILTER(LANG(?desc) = "en") }}
          OPTIONAL {{ ?item wdt:P577 ?date }}
          OPTIONAL {{ ?item p:P2047/psv:P2047 [wikibase:quantityAmount ?amount;
                                                wikibase:quantityUnit ?unit] }}
          OPTIONAL {{ ?item wdt:P57 ?dir. ?dir rdfs:label ?dirLabel
                     FILTER(LANG(?dirLabel) = "en") }}
        }}""")
    films = {}
    for r in rows:
        f = films.setdefault(r["item"], Film(qid(r["item"]), r.get("label", qid(r["item"]))))
        f.rank = int(r["ordinal"])
        f.description = r.get("desc", "")
        if "date" in r:
            year = int(r["date"][:4])
            f.year = min(f.year or year, year)
        if "amount" in r and qid(r["unit"]) in UNIT_MINUTES:
            f.minutes = float(r["amount"]) * UNIT_MINUTES[qid(r["unit"])]
        if "dirLabel" in r and r["dirLabel"] not in f.directors:
            f.directors.append(r["dirLabel"])

    def runtime_fit(f):
        if not runtime_secs or not f.minutes:
            return 1
        return 0 if runtime_matches(f, runtime_secs) else 2

    return sorted(films.values(), key=lambda f: (runtime_fit(f), f.rank))


def runtime_matches(film, runtime_secs):
    # PAL discs run ~4% fast, and listed runtimes are often rounded
    diff = abs(film.minutes - runtime_secs / 60)
    return diff <= max(5, film.minutes * 0.06)


def confident(candidates, name, runtime_secs):
    """The candidate to use without asking, or None if it's a guess."""
    if not candidates:
        return None
    best = candidates[0]
    if best.minutes and runtime_matches(best, runtime_secs):
        return best
    exact = [c for c in candidates if c.title.casefold() == name.casefold()]
    return exact[0] if len(exact) == 1 else None


# ---------------------------------------------------------------- details

def details(item_qid):
    rows = sparql(f"""
        SELECT ?prop ?value ?label ?extra WHERE {{
          VALUES ?item {{ wd:{item_qid} }}
          {{ ?item rdfs:label ?value FILTER(LANG(?value) = "en") BIND("title" AS ?prop) }}
          UNION {{ ?item schema:description ?value FILTER(LANG(?value) = "en")
                   BIND("description" AS ?prop) }}
          UNION {{ ?item wdt:P1476 ?value BIND("original_title" AS ?prop) }}
          UNION {{ ?item p:P577/psv:P577 [wikibase:timeValue ?value;
                                          wikibase:timePrecision ?extra]
                   BIND("date" AS ?prop) }}
          UNION {{ ?item p:P2047/psv:P2047 [wikibase:quantityAmount ?value;
                                            wikibase:quantityUnit ?extra]
                   BIND("duration" AS ?prop) }}
          UNION {{ ?item wdt:P57 ?value BIND("directors" AS ?prop) }}
          UNION {{ ?item wdt:P58 ?value BIND("writers" AS ?prop) }}
          UNION {{ ?item wdt:P161 ?value BIND("cast" AS ?prop) }}
          UNION {{ ?item wdt:P136 ?value BIND("genres" AS ?prop) }}
          UNION {{ ?item wdt:P495 ?value BIND("countries" AS ?prop) }}
          UNION {{ ?item wdt:P272 ?value BIND("studios" AS ?prop) }}
          UNION {{ ?item wdt:P364/wdt:P219 ?value BIND("languages" AS ?prop) }}
          UNION {{ ?item wdt:P345 ?value BIND("imdb" AS ?prop) }}
          UNION {{ ?item wdt:P4947 ?value BIND("tmdb" AS ?prop) }}
          UNION {{ ?item wdt:P3383 ?value BIND("poster_url" AS ?prop) }}
          UNION {{ ?value schema:about ?item; schema:isPartOf <https://en.wikipedia.org/>
                   BIND("enwiki" AS ?prop) }}
          OPTIONAL {{ ?value rdfs:label ?label FILTER(LANG(?label) = "en") }}
        }}""")
    film = Film(item_qid, item_qid)
    dates = []
    for r in rows:
        prop, value = r["prop"], r.get("label") or r["value"]
        if prop == "date":
            dates.append((r["value"][:10], int(r["extra"])))
        elif prop == "duration":
            unit = qid(r["extra"])
            if unit in UNIT_MINUTES:
                film.minutes = float(r["value"]) * UNIT_MINUTES[unit]
        elif prop == "genres":
            value = re.sub(r" film$", "", value)
            value = value[:1].upper() + value[1:]
            if value not in film.genres:
                film.genres.append(value)
        elif isinstance(getattr(film, prop), list):
            if value not in getattr(film, prop):
                getattr(film, prop).append(value)
        else:
            setattr(film, prop, value)
    if dates:
        day, precision = min(dates)
        film.year = int(day[:4])
        if precision >= 11:  # 11 = day
            film.premiered = day
    if film.original_title == film.title:
        film.original_title = None
    film.plot = wikipedia_summary(film.enwiki)
    return film


def wikipedia_summary(article_url):
    if not article_url:
        return None
    title = article_url.rsplit("/", 1)[-1]
    try:
        return get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}").get("extract")
    except Exception:
        return None


# ---------------------------------------------------------------- output

def write_nfo(film, path, source="DVD"):
    root = ET.Element("movie")

    def add(tag, text, **attrs):
        if text:
            el = ET.SubElement(root, tag, attrs)
            el.text = str(text)
            return el

    add("title", film.title)
    add("originaltitle", film.original_title)
    add("year", film.year)
    add("premiered", film.premiered)
    add("runtime", round(film.minutes) if film.minutes else None)
    add("plot", film.plot)
    for tag, values in [("genre", film.genres), ("country", film.countries),
                        ("director", film.directors), ("credits", film.writers),
                        ("studio", film.studios)]:
        for v in values:
            add(tag, v)
    for i, name in enumerate(film.cast):
        actor = ET.SubElement(root, "actor")
        ET.SubElement(actor, "name").text = name
        ET.SubElement(actor, "order").text = str(i)
    add("uniqueid", film.imdb, type="imdb", default="true")
    add("uniqueid", film.tmdb, type="tmdb")
    add("uniqueid", film.qid, type="wikidata")
    add("source", source)
    ET.indent(root)
    path.write_text('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                    + ET.tostring(root, encoding="unicode") + "\n")


def save_artwork(film, folder):
    """Write poster.jpg (and fanart.jpg from TMDb). Returns the files written."""
    written = []
    urls = {}
    key = os.environ.get("TMDB_API_KEY")
    if key and film.tmdb:
        try:
            info = get(f"https://api.themoviedb.org/3/movie/{film.tmdb}", {"api_key": key})
            for name, field_ in [("poster", "poster_path"), ("fanart", "backdrop_path")]:
                if info.get(field_):
                    urls[name] = "https://image.tmdb.org/t/p/original" + info[field_]
            film.plot = film.plot or info.get("overview")
        except Exception as e:
            print(f"  TMDb lookup failed: {e}")
    if "poster" not in urls and film.poster_url:
        urls["poster"] = film.poster_url + "?width=1000"  # Commons file
    for name, url in urls.items():
        try:
            body, ctype = get(url, raw=True)
        except Exception as e:
            print(f"  couldn't fetch {name}: {e}")
            continue
        ext = {"image/png": ".png", "image/webp": ".webp"}.get(ctype, ".jpg")
        path = folder / f"{name}{ext}"
        path.write_bytes(body)
        written.append(path)
    return written
