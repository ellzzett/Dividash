"""
Dividenden-Screener mit Punkteschema (max. 27 Punkte)

Kategorien (je 0-3 Punkte):
  1. Durchschnittliche Dividendenrendite 10 Jahre
  2. Payout-Ratio
  3. Jahre ohne Dividendenkürzung
  4. Anzahl aufeinanderfolgender Dividendenanhebungen
  5. Aktuelle Dividendenrendite
  6. Dividendenwachstum 5 Jahre (CAGR)
  7. Eigenkapitalrendite (ROE)
  8. Verschuldung (Debt/Equity, Finanzsektor ausgenommen)
  9. Dividendendeckung durch den freien Cashflow

Installation:  pip install yfinance pandas lxml requests

Ohne Optionen wird ALLES geprüft: S&P 500, DAX, MDAX + iShares Europa, Schwellenländer,
Pazifik und Welt (mehrere tausend Aktien, der erste Lauf dauert Stunden, danach hilft der Cache).

Beispiele (in Colab mit "!" davor):
  python dividenden_screener.py                                  # alles (Standard)
  python dividenden_screener.py --universum sp500                # nur S&P 500
  python dividenden_screener.py --universum sp500 dax mdax       # Wikipedia-Indizes
  python dividenden_screener.py --region europa                  # alle europäischen Aktien
  python dividenden_screener.py --region asien afrika suedamerika
  python dividenden_screener.py --ishares acwi                   # MSCI ACWI (weltweit, Large/Mid)
  python dividenden_screener.py --ishares meine_holdings.csv     # manuell geladene iShares-CSV
  python dividenden_screener.py --datei meine_ticker.txt         # eigene Liste, ein Ticker pro Zeile

Optionen: --top N  --min-punkte N  --limit N (Test)  --workers N  --cache DATEI  --cache-tage N
"""
import argparse
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

import pandas as pd

# ================================================================ Schwellen
# Jede Liste: (Schwelle, Punkte), von streng nach locker. Erster Treffer gilt.
RENDITE_10J = [(0.03, 3), (0.02, 2), (0.01, 1)]            # strikt >
RENDITE_AKT = [(0.05, 3), (0.035, 2), (0.02, 1)]           # strikt >
WACHSTUM_5J = [(0.10, 3), (0.05, 2), (0.0, 1)]             # strikt >
KEINE_KUERZUNG = [(10, 3), (8, 2), (5, 1)]                 # Jahre, >=
ANHEBUNGEN = [(20, 3), (10, 2), (3, 1)]                    # Anzahl in Folge, >=
PAYOUT_BAENDER = [((0.25, 0.75), 3), ((0.20, 0.80), 2), ((0.15, 0.85), 1)]  # inklusive

ROE_STUFEN = [(0.15, 3), (0.10, 2), (0.05, 1)]             # strikt >
SCHULDEN_STUFEN = [(50, 3), (100, 2), (150, 1)]            # Debt/Equity in %, strikt <, nicht negativ
FCF_PAYOUT_STUFEN = [(0.60, 3), (0.80, 2), (1.00, 1)]      # Dividenden / Free Cashflow, <=, nur > 0
SEKTOR_OHNE_SCHULDEN = "Financial Services"                # Banken/Versicherer: Verschuldung nicht bewertet

KUERZUNG_TOLERANZ = 0.02   # Rückgang unter 2 % gilt nicht als Kürzung (Währung, Rundung)
ANHEBUNG_MIN = 0.005       # Anstieg ab 0,5 % gilt als Anhebung
SONDER_FAKTOR = 2.0        # Zahlung > 2x Median (bei >= 3 Zahlungen im Jahr) = Sonderdividende
WARN_RENDITE = 0.15        # Renditen darüber werden als "prüfen" markiert

# Ohne Optionen wird dieses Universum geprüft (alles, Duplikate werden entfernt)
STANDARD_UNIVERSEN = ["sp500", "dax", "mdax"]
STANDARD_ISHARES = ["ieur", "iemg", "ipac", "acwi"]   # Europa, Schwellenländer, Pazifik, Welt (u. a. Kanada)

CACHE_VERSION = 5
CACHE_DATEI = "screener_cache.json"


# ============================================================ Punktevergabe
def punkte_ueber(wert, stufen, inklusiv=False):
    if wert is None or pd.isna(wert):
        return 0
    for schwelle, p in stufen:
        if (wert >= schwelle) if inklusiv else (wert > schwelle):
            return p
    return 0


def punkte_payout(payout):
    if payout is None or pd.isna(payout):
        return 0
    for (lo, hi), p in PAYOUT_BAENDER:
        if lo <= payout <= hi:
            return p
    return 0


def punkte_unter(wert, stufen, inklusiv=False):
    """Kleiner ist besser. Negative Werte (z. B. negatives Eigenkapital) geben 0 Punkte."""
    if wert is None or pd.isna(wert) or wert < 0:
        return 0
    for schwelle, p in stufen:
        if (wert <= schwelle) if inklusiv else (wert < schwelle):
            return p
    return 0


def bewerte(k: dict) -> dict:
    p = {
        "p_rendite_10j": punkte_ueber(k["rendite_10j"], RENDITE_10J),
        "p_payout": punkte_payout(k["payout"]),
        "p_keine_kuerzung": punkte_ueber(k["jahre_ohne_kuerzung"], KEINE_KUERZUNG, inklusiv=True),
        "p_anhebungen": punkte_ueber(k["anhebungen"], ANHEBUNGEN, inklusiv=True),
        "p_rendite_aktuell": punkte_ueber(k["rendite_aktuell"], RENDITE_AKT),
        "p_wachstum_5j": punkte_ueber(k["wachstum_5j"], WACHSTUM_5J),
        "p_roe": punkte_ueber(k.get("roe"), ROE_STUFEN),
        "p_schulden": 0 if k.get("sektor") == SEKTOR_OHNE_SCHULDEN
                      else punkte_unter(k.get("schulden"), SCHULDEN_STUFEN),
        "p_fcf": (punkte_unter(k.get("fcf_payout"), FCF_PAYOUT_STUFEN, inklusiv=True)
                  if (k.get("fcf_payout") or 0) > 0 else 0),
    }
    p["punkte_gesamt"] = sum(p.values())
    return p


# ================================================================ Kennzahlen
def _ohne_sonder(g: pd.Series) -> pd.Series:
    """Entfernt Sonderdividenden: Zahlungen > 2x Median (nur bei >= 3 Zahlungen)."""
    if len(g) >= 3:
        return g[g <= SONDER_FAKTOR * g.median()]
    return g


def _modus(werte):
    """Häufigster Wert, bei Gleichstand der kleinste (wie pandas .mode())."""
    z = {}
    for w in werte:
        z[w] = z.get(w, 0) + 1
    top = max(z.values())
    return min(w for w, c in z.items() if c == top)


def _verschiebe_luecken(listen: dict) -> dict:
    """
    Zahlungsverschiebung über ein komplett leeres Jahr: Hat ein Jahr doppelt so viele Zahlungen
    wie üblich (2m) und davor (oder danach) fehlt innerhalb der Historie ein Jahr ganz, werden die
    Zahlungen auf beide Jahre verteilt. Beispiel DNB: 2020 keine, 2021 zwei Zahlungen
    (Corona-Aufschub) -> je eine Zahlung für 2020 und 2021.
    """
    jahre = sorted(listen)
    n0 = {j: len(v) for j, v in listen.items()}
    m = _modus(n0.values())
    neu = dict(listen)
    for b in jahre:
        if n0[b] != 2 * m:
            continue
        vor, nach = b - 1, b + 1
        if vor > jahre[0] and vor not in listen and vor not in neu:
            neu[vor], neu[b] = listen[b][:m], listen[b][m:]
        elif nach < jahre[-1] and nach not in listen and nach not in neu:
            neu[nach], neu[b] = listen[b][m:], listen[b][:m]
    return neu


def jahressummen(dividenden: pd.Series, letztes_jahr: int) -> pd.Series:
    """
    Jahresdividende je Kalenderjahr (nur volle Jahre), bereinigt um
      - Sonderdividenden,
      - Zahlungsverschiebungen über ein leeres Jahr (siehe _verschiebe_luecken) und
      - Zahlungsverschiebungen über den Jahreswechsel (ein Jahr 5, das nächste 3
        statt 4+4 Zahlungen): beide Jahre werden auf die übliche Zahl normiert.
    """
    d = dividenden[dividenden > 0]
    listen = {}
    for jahr, g in d.groupby(d.index.year):
        if jahr > letztes_jahr:
            continue
        g = _ohne_sonder(g)
        if len(g):
            listen[int(jahr)] = [float(x) for x in g.values]
    if len(listen) >= 3:
        listen = _verschiebe_luecken(listen)
    summe = {j: float(sum(v)) for j, v in listen.items()}
    anzahl = {j: len(v) for j, v in listen.items()}
    s = pd.Series(summe, dtype=float).sort_index()
    if len(s) < 3:
        return s
    n = pd.Series(anzahl).sort_index()
    m = int(n.mode().iloc[0])
    jahre = list(s.index)
    for a, b in zip(jahre, jahre[1:]):
        if b == a + 1 and n.loc[a] != m and n.loc[a] + n.loc[b] == 2 * m:
            s.loc[a] = s.loc[a] * m / n.loc[a]
            s.loc[b] = s.loc[b] * m / n.loc[b]
            n.loc[a] = n.loc[b] = m
    return s


def kennzahlen(dividenden: pd.Series, kurse: pd.Series, payout, heute=None):
    """
    dividenden: Einzelzahlungen (DatetimeIndex), kurse: Schlusskurse (täglich),
    payout: Ausschüttungsquote als Dezimalzahl (0.55 = 55 %).
    """
    heute = heute or date.today()
    letztes = heute.year - 1
    js = jahressummen(dividenden, letztes)
    # Wer im letzten vollen Jahr nichts gezahlt hat, bekommt keine Serien-Punkte.
    aktiv = len(js) > 0 and js.index[-1] == letztes

    rendite_10j, ohne_kuerzung, anhebungen, wachstum_5j = None, 0, 0, None
    if aktiv:
        jahresend = kurse.groupby(kurse.index.year).last()
        soll = list(range(letztes - 9, letztes + 1))
        r = js.reindex(soll) / jahresend.reindex(soll)
        if r.notna().all():
            rendite_10j = float(r.mean())

        jahre = list(js.index)
        for i in range(len(jahre) - 1, 0, -1):
            neu, alt = js.loc[jahre[i]], js.loc[jahre[i - 1]]
            if jahre[i] - jahre[i - 1] != 1 or neu < alt * (1 - KUERZUNG_TOLERANZ):
                break
            ohne_kuerzung += 1
        for i in range(len(jahre) - 1, 0, -1):
            neu, alt = js.loc[jahre[i]], js.loc[jahre[i - 1]]
            if jahre[i] - jahre[i - 1] != 1 or neu <= alt * (1 + ANHEBUNG_MIN):
                break
            anhebungen += 1

        if (letztes - 5) in js.index and js.loc[letztes - 5] > 0:
            wachstum_5j = float((js.loc[letztes] / js.loc[letztes - 5]) ** (1 / 5) - 1)

    # Aktuelle Rendite: Dividenden der letzten 12 Monate (ohne Sonderzahlungen) / letzter Kurs
    idx = dividenden.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    ab = pd.Timestamp(heute) - pd.DateOffset(years=1)
    d12 = _ohne_sonder(dividenden[(idx > ab) & (dividenden.values > 0)]).sum()
    rendite_akt = float(d12 / kurse.iloc[-1]) if len(kurse) and kurse.iloc[-1] > 0 else None

    return {
        "rendite_10j": rendite_10j,
        "payout": payout,
        "jahre_ohne_kuerzung": ohne_kuerzung,
        "anhebungen": anhebungen,
        "rendite_aktuell": rendite_akt,
        "wachstum_5j": wachstum_5j,
        "roe": None,
        "schulden": None,
        "fcf_payout": None,
    }


def rohdaten(dividenden: pd.Series, kurse: pd.Series, heute=None) -> dict:
    """
    Rohdaten für das Dashboard: Mit ihnen berechnet das Dashboard die Kennzahlen im Browser
    neu, sobald Toleranzen (Kürzung, Anhebung, Sonderdividende) verändert werden.
    """
    heute = heute or date.today()
    letztes = heute.year - 1
    d = dividenden[dividenden > 0]
    zahlungen = {}
    for jahr, g in d.groupby(d.index.year):
        zahlungen[str(int(jahr))] = [float(f"{x:.5g}") for x in g.values]
    idx = d.index
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    ab = pd.Timestamp(heute) - pd.DateOffset(years=1)
    z12 = [float(f"{x:.5g}") for x in d.values[idx > ab]]
    jahresend = kurse.groupby(kurse.index.year).last()
    jahresend = {str(int(j)): float(f"{v:.5g}") for j, v in jahresend.items()
                 if letztes - 10 <= j <= letztes}
    return {"zahlungen": zahlungen, "zahlungen_12m": z12, "jahresend": jahresend,
            "kurs": float(f"{kurse.iloc[-1]:.5g}")}


def warnhinweis(k: dict) -> str:
    w = []
    if k["rendite_aktuell"] and k["rendite_aktuell"] > WARN_RENDITE:
        w.append("aktuelle Rendite > 15 % (Datenfehler/Dividendenfalle?)")
    if k["rendite_10j"] and k["rendite_10j"] > WARN_RENDITE:
        w.append("10-Jahres-Rendite > 15 % (Datenfehler?)")
    if k["payout"] and k["payout"] > 1:
        w.append("Payout > 100 %")
    return "; ".join(w)


# ====================================================== Universum: Wikipedia
# name: (Wikipedia-Seite, Börsen-Suffix für Yahoo, "." im Ticker durch "-" ersetzen)
UNIVERSEN = {
    "sp500": ("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", "", True),
    "aristocrats": ("https://en.wikipedia.org/wiki/S%26P_500_Dividend_Aristocrats", "", True),
    "dax": ("https://en.wikipedia.org/wiki/DAX", ".DE", False),
    "mdax": ("https://en.wikipedia.org/wiki/MDAX", ".DE", False),
}


def ticker_aus_tabellen(tabellen):
    """Sucht die erste Tabelle mit einer Ticker-/Symbol-Spalte (>= 20 Zeilen)."""
    for df in tabellen:
        for c in df.columns:
            name = " ".join(map(str, c)) if isinstance(c, tuple) else str(c)
            if "ticker" in name.lower() or "symbol" in name.lower():
                werte = df[c].dropna().astype(str)
                werte = [re.sub(r"\[.*?\]", "", w).strip().upper() for w in werte]
                werte = [w for w in werte if w and w != "NAN"]
                if len(werte) >= 20:
                    return werte
    raise ValueError("Keine Ticker-Spalte gefunden")


def lade_universum(name):
    import requests

    url, suffix, punkt_zu_strich = UNIVERSEN[name]
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (dividenden-screener)"},
                        timeout=30).text
    out = []
    for t in ticker_aus_tabellen(pd.read_html(io.StringIO(html))):
        if punkt_zu_strich:
            t = t.replace(".", "-")
        if suffix and "." not in t:
            t += suffix
        out.append(t)
    return out


# ================================================ Universum: ETF-Holdings (global)
# iShares veröffentlicht die Holdings seiner ETFs als CSV (Ticker, Börse, Land).
# Produkt-IDs (geprüft): ACWI 239600, IEMG 244050, IEUR 264617, IPAC 264619
ISHARES_ETFS = {
    "acwi": 239600,   # MSCI ACWI: weltweit, Large/Mid Cap (~2.300 Aktien)
    "iemg": 244050,   # Core MSCI Emerging Markets IMI (~2.700): Asien, Lateinamerika, Afrika, Nahost
    "ieur": 264617,   # Core MSCI Europe IMI (~1.000): Europa inkl. Small Caps
    "ipac": 264619,   # Core MSCI Pacific IMI (~1.400): Japan, Australien, Hongkong, Singapur, NZ
}

# Region -> Länder-Stichworte (Kleinbuchstaben, Teilstring der Spalte "Location")
REGIONEN = {
    "europa": ["austria", "belgium", "denmark", "finland", "france", "germany", "ireland", "italy",
               "netherlands", "norway", "portugal", "spain", "sweden", "switzerland",
               "united kingdom", "luxembourg", "greece", "poland", "czech", "hungary", "turkey"],
    "asien": ["japan", "china", "hong kong", "taiwan", "korea", "india", "singapore", "indonesia",
              "malaysia", "thailand", "philippines", "australia", "new zealand"],
    "afrika": ["south africa", "egypt", "morocco", "nigeria", "kenya"],
    "suedamerika": ["brazil", "chile", "colombia", "peru", "argentina"],
    "nahost": ["saudi", "united arab emirates", "qatar", "kuwait", "israel"],
}
# Welche ETFs decken eine Region ab?
REGION_QUELLEN = {
    "europa": ["ieur"],
    "asien": ["iemg", "ipac"],
    "afrika": ["iemg"],
    "suedamerika": ["iemg"],
    "nahost": ["iemg"],
}

# Börsenname (Kleinbuchstaben, Teilstring) -> Yahoo-Suffix. Die Zuordnung ist nicht
# vollständig geprüft: Unbekannte Börsen werden übersprungen und am Ende aufgelistet.
BOERSEN = {
    "nasdaq": "", "new york stock exchange": "", "nyse": "", "cboe": "",
    "london stock exchange": ".L", "xetra": ".DE", "tokyo": ".T", "toronto": ".TO",
    "euronext paris": ".PA", "euronext amsterdam": ".AS", "euronext brussels": ".BR",
    "euronext lisbon": ".LS", "euronext dublin": ".IR", "irish": ".IR",
    "borsa italiana": ".MI", "madrid": ".MC", "six swiss": ".SW", "oslo": ".OL",
    "wiener": ".VI", "warsaw": ".WA", "prague": ".PR", "budapest": ".BD", "athens": ".AT",
    "istanbul": ".IS",
    "australian": ".AX", "asx": ".AX", "helsinki": ".HE", "hong kong": ".HK", "gretai": ".TWO", "taiwan": ".TW",
    "kosdaq": ".KQ", "korea": ".KS", "singapore": ".SI",
    "national stock exchange of india": ".NS", "shanghai": ".SS", "shenzhen": ".SZ",
    "new zealand": ".NZ", "thailand": ".BK", "bursa malaysia": ".KL", "indonesia": ".JK",
    "philippine": ".PS",
    "johannesburg": ".JO", "egypt": ".CA",
    "b3": ".SA", "brasil": ".SA", "bovespa": ".SA", "bolsa mexicana": ".MX", "santiago": ".SN", "colombia": ".CL",
    "buenos aires": ".BA",
    "tel aviv": ".TA", "saudi": ".SR", "qatar": ".QA", "kuwait": ".KW",
}
NORDIC = {"sweden": ".ST", "denmark": ".CO", "finland": ".HE"}


# Börsen, die in ihrem Namen "nyse"/"nasdaq" enthalten können, aber keine US-Börsen sind
# (z. B. "NYSE Euronext - Euronext Paris", "Nasdaq OMX Helsinki"). Deshalb werden diese
# US-Schlüssel erst NACH allen anderen Börsen geprüft.
US_BOERSEN = ("nasdaq", "new york stock exchange", "nyse", "cboe")

# Fallback über das Land, wenn der Börsenname nicht erkannt wird
LAND_SUFFIX = {
    "france": ".PA", "belgium": ".BR", "portugal": ".LS", "netherlands": ".AS",
    "ireland": ".IR", "italy": ".MI", "spain": ".MC", "norway": ".OL", "austria": ".VI",
    "finland": ".HE", "sweden": ".ST", "denmark": ".CO", "greece": ".AT", "poland": ".WA",
    "switzerland": ".SW", "germany": ".DE", "united kingdom": ".L", "australia": ".AX",
    "brazil": ".SA", "japan": ".T", "canada": ".TO", "new zealand": ".NZ", "israel": ".TA",
    "singapore": ".SI", "hong kong": ".HK", "south africa": ".JO", "mexico": ".MX",
}


def yahoo_ticker(ticker, exchange, location):
    t = str(ticker).strip().upper().replace("*", "").replace(" ", "-")
    if t in ("", "-", "NAN"):
        return None
    ex = str(exchange).lower()
    land = str(location).lower()
    suffix = None
    if "nordic" in ex or "omx" in ex:
        suffix = NORDIC.get(land)
    if suffix is None:
        for key, suf in BOERSEN.items():
            if key not in US_BOERSEN and key in ex:
                suffix = suf
                break
    if suffix is None and any(k in ex for k in US_BOERSEN) and "euronext" not in ex:
        suffix = ""
    if suffix is None:
        suffix = LAND_SUFFIX.get(land)
    if suffix is None:
        return None
    if suffix == "":
        return t.replace(".", "-")
    if suffix == ".L":
        t = t.rstrip(".").replace(".", "-")
    elif suffix == ".HK":
        t = t.lstrip("0").zfill(4)
    elif suffix in (".SS", ".SZ", ".KS", ".KQ"):
        t = t.zfill(6)
    return t + suffix


def _ishares_text(quelle):
    import requests

    if quelle.lower() in ISHARES_ETFS:
        pid = ISHARES_ETFS[quelle.lower()]
        urls = [f"https://www.ishares.com/us/products/{pid}/x/latest-holdings.csv",
                f"https://www.ishares.com/us/products/{pid}/x/1467271812596.ajax"
                f"?fileType=csv&fileName=holdings&dataType=fund"]
    elif quelle.startswith("http"):
        urls = [quelle]
    else:
        with open(quelle, encoding="utf-8-sig") as f:
            return f.read()
    for u in urls:
        try:
            r = requests.get(u, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            r.raise_for_status()
            text = r.content.decode("utf-8-sig", errors="replace")
            if "Asset Class" in text and "Ticker" in text:
                return text
        except Exception:
            continue
    raise RuntimeError("Download fehlgeschlagen. Lade die CSV auf ishares.com (ETF > Holdings > "
                       "Download) und übergib den Dateipfad: --ishares datei.csv")


def lade_ishares(quelle, regionen=None):
    """quelle: Kürzel (acwi, iemg, ieur, ipac), URL oder lokale CSV-Datei."""
    zeilen = _ishares_text(quelle).splitlines()
    start = next(i for i, z in enumerate(zeilen) if z.strip('"\ufeff ').startswith("Ticker"))
    block = []
    for z in zeilen[start:]:
        if not z.strip():
            break  # danach folgt nur Rechtstext
        block.append(z)
    df = pd.read_csv(io.StringIO("\n".join(block)), dtype=str, on_bad_lines="skip")
    df = df[df["Asset Class"] == "Equity"]
    if regionen:
        stichworte = [w for r in regionen for w in REGIONEN[r]]
        loc = df["Location"].fillna("").str.lower()
        df = df[loc.apply(lambda x: any(w in x for w in stichworte))]
    out, unbekannt = [], {}
    for _, r in df.iterrows():
        y = yahoo_ticker(r["Ticker"], r["Exchange"], r["Location"])
        if y:
            out.append(y)
        else:
            unbekannt[r["Exchange"]] = unbekannt.get(r["Exchange"], 0) + 1
    if unbekannt:
        print("  Übersprungen (Börse unbekannt):",
              ", ".join(f"{k} ({v})" for k, v in sorted(unbekannt.items(), key=lambda x: -x[1])))
    return out


def lade_alle(namen=(), datei=None, ishares=(), regionen=None):
    ticker = []
    for n in namen:
        try:
            liste = lade_universum(n)
            print(f"{n}: {len(liste)} Ticker geladen")
            ticker += liste
        except Exception as e:
            print(f"{n}: konnte nicht geladen werden ({e})")
    for q in ishares:
        try:
            liste = lade_ishares(q, regionen)
            print(f"iShares {q}: {len(liste)} Ticker geladen")
            ticker += liste
        except Exception as e:
            print(f"iShares {q}: konnte nicht geladen werden ({e})")
    if datei:
        with open(datei, encoding="utf-8") as f:
            ticker += [z.strip() for z in f if z.strip() and not z.startswith("#")]
    return list(dict.fromkeys(ticker))  # Duplikate entfernen, Reihenfolge behalten


# ================================================================== Datenabruf
def analysiere(yf, t, min_punkte):
    tk = yf.Ticker(t)
    # Ein einziger Abruf liefert Kurse und Dividenden. Unbereinigte Kurse (nur Splits),
    # sonst wären historische Renditen verzerrt.
    h = tk.history(period="max", auto_adjust=False, actions=True)
    if h is None or h.empty or "Close" not in h:
        return None
    kurse = h["Close"].dropna()
    div = h["Dividends"] if "Dividends" in h else tk.dividends
    div = div[div > 0]
    if div.empty or kurse.empty:
        return None

    k = kennzahlen(div, kurse, None)
    punkte = bewerte(k)
    name = sektor = waehrung = None
    geprueft = False
    # Der langsame .info-Abruf lohnt nur, wenn die Mindestpunktzahl noch erreichbar ist (+12 max.)
    if punkte["punkte_gesamt"] + 12 >= min_punkte:
        info = tk.info or {}
        if len(info) > 5:  # leere/gedrosselte Antworten nicht als "geprüft" speichern
            k["payout"] = info.get("payoutRatio")
            k["roe"] = info.get("returnOnEquity")
            k["schulden"] = info.get("debtToEquity")      # Yahoo liefert Prozent (45 = 45 %)
            k["sektor"] = info.get("sector")
            # Dividendendeckung: Dividenden (= Payout x Gewinn) / Free Cashflow. Beide Größen
            # stehen in der Berichtswährung, so entstehen keine Währungsfehler.
            fcf, gewinn, po = (info.get("freeCashflow"), info.get("netIncomeToCommon"),
                               info.get("payoutRatio"))
            if all(isinstance(x, (int, float)) for x in (fcf, gewinn, po)) \
                    and fcf > 0 and gewinn > 0 and po > 0:
                k["fcf_payout"] = float(po * gewinn / fcf)
            name = info.get("longName") or info.get("shortName")
            sektor, waehrung = info.get("sector"), info.get("currency")
            punkte = bewerte(k)
            geprueft = True
    return {"ticker": t, "name": name, "sektor": sektor, "waehrung": waehrung,
            **k, **punkte, "warnung": warnhinweis(k), "payout_geprueft": geprueft,
            **rohdaten(div, kurse)}


def _lade_cache(pfad):
    try:
        with open(pfad, encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("_version") == CACHE_VERSION:
            return cache
    except Exception:
        pass
    return {"_version": CACHE_VERSION}


def _speichere_cache(cache, pfad):
    with open(pfad, "w", encoding="utf-8") as f:
        json.dump(cache, f, default=float, separators=(",", ":"))


def screene(tickers, min_punkte=0, workers=4, cache_tage=7, cache_datei=None):
    import yfinance as yf

    cache_datei = cache_datei or CACHE_DATEI
    cache = _lade_cache(cache_datei)
    jetzt = time.time()
    zeilen, offen = [], []
    for t in tickers:
        c = cache.get(t)
        if c and jetzt - c["zeit"] < cache_tage * 86400:
            row = c["row"]
            if row is None:
                continue  # zahlt keine Dividende
            if row["payout_geprueft"] or row["punkte_gesamt"] + 12 < min_punkte:
                zeilen.append(row)
                continue
        offen.append(t)

    def job(t):
        for _ in range(2):
            try:
                return t, analysiere(yf, t, min_punkte)
            except Exception:
                time.sleep(2)
        return t, "fehler"

    print(f"{len(offen)} Ticker werden abgerufen, {len(tickers) - len(offen)} aus dem Cache")
    fehler = 0
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(job, t) for t in offen]
        for i, f in enumerate(as_completed(futs), 1):
            t, row = f.result()
            if isinstance(row, str):
                fehler += 1  # wird beim nächsten Lauf erneut versucht
            else:
                cache[t] = {"zeit": jetzt, "row": row}
                if row:
                    zeilen.append(row)
            if i % 25 == 0:
                print(f"  {i}/{len(offen)} geprüft")
                _speichere_cache(cache, cache_datei)
    _speichere_cache(cache, cache_datei)
    if fehler:
        print(f"{fehler} Ticker mit Abruffehler (erneuter Lauf versucht sie nochmal)")

    df = pd.DataFrame(zeilen)
    if df.empty:
        return df
    df["name"] = df["name"].fillna(df["ticker"])
    return df.sort_values(["punkte_gesamt", "rendite_aktuell"],
                          ascending=False).reset_index(drop=True)


# ======================================================================== Start
def main():
    ap = argparse.ArgumentParser(description="Dividenden-Screener")
    ap.add_argument("--universum", nargs="+", choices=list(UNIVERSEN),
                    help="Wikipedia-Indizes: " + ", ".join(UNIVERSEN))
    ap.add_argument("--ishares", nargs="+",
                    help="ETF-Kürzel (acwi, iemg, ieur, ipac), URL oder CSV-Datei")
    ap.add_argument("--region", nargs="+", choices=list(REGIONEN),
                    help="Länder-Filter für iShares-Listen; wählt passende ETFs automatisch")
    ap.add_argument("--datei", help="Textdatei mit eigenen Tickern")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--min-punkte", type=int, default=12)
    ap.add_argument("--limit", type=int, help="nur die ersten N Ticker (Test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default=CACHE_DATEI, help="Pfad der Cache-Datei")
    ap.add_argument("--cache-tage", type=int, default=7)
    args, _ = ap.parse_known_args()

    keine_auswahl = not (args.universum or args.ishares or args.region or args.datei)
    if keine_auswahl:
        universen, ishares = list(STANDARD_UNIVERSEN), list(STANDARD_ISHARES)
        print("Keine Optionen angegeben: prüfe das gesamte Standard-Universum "
              f"({', '.join(universen + ishares)})")
    else:
        ishares = list(args.ishares or [])
        if args.region and not ishares:
            for r in args.region:
                ishares += [q for q in REGION_QUELLEN[r] if q not in ishares]
        universen = args.universum or ([] if (args.datei or ishares) else ["sp500"])

    ticker = lade_alle(universen, args.datei, ishares, args.region)
    if args.limit:
        ticker = ticker[:args.limit]
    print(f"{len(ticker)} Ticker im Universum")
    if not ticker:
        return

    alle = screene(ticker, args.min_punkte, args.workers, args.cache_tage, args.cache)
    if alle.empty:
        print("Keine Dividendenaktie gefunden.")
        return

    # Rohkennzahlen ALLER Dividendenaktien für das Dashboard (Punkte rechnet das Dashboard selbst)
    os.makedirs("docs", exist_ok=True)
    roh = alle.drop(columns=[c for c in alle.columns
                             if c.startswith("p_") or c == "punkte_gesamt"])
    roh = roh.astype(object).where(roh.notna(), None)
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump({"updated": date.today().isoformat(),
                   "rows": roh.to_dict("records")}, f, ensure_ascii=False, default=float, separators=(",", ":"))

    ergebnis = alle[alle.punkte_gesamt >= args.min_punkte].reset_index(drop=True)
    if ergebnis.empty:
        print("Keine Aktie erreicht die Mindestpunktzahl.")
        return
    roh_spalten = ["zahlungen", "zahlungen_12m", "jahresend"]
    ergebnis.drop(columns=[c for c in roh_spalten if c in ergebnis.columns]) \
        .to_csv("screener_ergebnis.csv", index=False)

    spalten = [c for c in ["ticker", "name", "sektor", "punkte_gesamt", "rendite_aktuell",
                           "rendite_10j", "payout", "jahre_ohne_kuerzung", "anhebungen",
                           "wachstum_5j", "roe", "schulden", "fcf_payout", "warnung"] if c in ergebnis.columns]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(f"\nTop {args.top} von {len(ergebnis)} Treffern (max. 27 Punkte), "
          f"vollständige Liste: screener_ergebnis.csv\n")
    print(ergebnis[spalten].head(args.top).round(3).to_string())


if __name__ == "__main__":
    main()
