"""
Kicktipp Bundesliga Tipp-Assistent v2 - alles in einer Datei fuer minimalen
Setup-Aufwand (3 Dateien insgesamt: diese hier, requirements.txt, Workflow).

NEU GEGENUEBER V1 (nach der Auswertung von Spieltag 1 2026/27):
 - Negative Binomial statt reinem Poisson waehlbar (USE_NEGATIVE_BINOMIAL) -
   bildet Kantersiege realistischer ab, r wird mitgefittet statt geraten.
 - Kopf-an-Kopf-Gewichtung: direkte Historie zwischen den beiden konkreten
   Teams fliesst zusaetzlich zur generischen Team-Staerke ein.
 - Korrekte Kicktipp-Punkteregel (2/3/5, kein Tordifferenz-Zwischenschritt
   bei Unentschieden) - war in v1 falsch (4/3/2 angenommen).
 - Realistischerer Aufsteiger-Fallback statt reinem Liga-Durchschnitt.

Ablauf bei jedem automatischen Lauf:
 1. Ergebnisse von OpenLigaDB laden (mehrere Saisons)
 2. Modell neu fitten (Dixon-Coles, Poisson oder Negative Binomial)
 3. Fuer jedes anstehende Spiel: generische Vorhersage -> mit H2H-Historie
    anpassen -> mit Marktquoten anpassen (falls ODDS_API_KEY gesetzt)
 4. Tipp mit hoechster erwarteter Kicktipp-Punktzahl waehlen
 5. Log-Datei + Homescreen-Seite (docs/index.html) schreiben, E-Mail schicken
"""
import os
import re
import smtplib
import unicodedata
from datetime import date, datetime, timezone
from difflib import SequenceMatcher
from email.mime.text import MIMEText

import numpy as np
import requests
from scipy.optimize import minimize
from scipy.stats import poisson, nbinom

# ============================================================
# KONFIGURATION - hier darfst du gerne dran schrauben
# ============================================================
LEAGUE_SHORTCUT = "bl1"                       # "bl2" fuer die 2. Bundesliga
USE_NEGATIVE_BINOMIAL = True                  # s. Docstring oben; per Backtest ueberpruefbar
ODDS_WEIGHT = 0.5                             # 0 = nur Modell, 1 = nur Markt
H2H_MIN_DUELLE = 3                            # ab wie vielen Duellen H2H ueberhaupt einfliesst
H2H_MAX_DUELLE = 10                           # hoechstens die juengsten X Duelle beruecksichtigen
PUNKTE_ERGEBNIS = 5                           # Kicktipp-Punkte: exaktes Ergebnis (echte Regel: 2-3-5)
PUNKTE_TORDIFFERENZ = 3                       # ... nur bei Sieg: richtige Tordifferenz
PUNKTE_TENDENZ = 2                            # ... Sieger/Unentschieden richtig, sonst nichts
MAX_GOALS = 8                                 # betrachtete Tore je Team (0..8)
AUFSTEIGER_ATTACK_DEFAULT = -0.10             # Fallback-Staerke fuer Teams ohne Erstliga-Historie
AUFSTEIGER_DEFENSE_DEFAULT = 0.10

BASE_URL_LIGA = "https://api.openligadb.de"
BASE_URL_ODDS = "https://api.the-odds-api.com/v4"
SPORT_KEY_ODDS = "soccer_germany_bundesliga"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TIPPS_DIR = os.path.join(SCRIPT_DIR, "tipps")
DOCS_DIR = os.path.join(SCRIPT_DIR, "docs")

_heute = date.today()
_aktuelle_saison = _heute.year if _heute.month >= 7 else _heute.year - 1
# Zwei Saisons als Trainingsfenster - fuer eigenstaendige Backtests (backtest.py)
# wird dieselbe Funktion mit anderen Saisons aufgerufen.
TRAININGS_SAISONS = [_aktuelle_saison, _aktuelle_saison - 1]


# ============================================================
# OPENLIGADB - Ergebnisse holen (kein API-Key noetig)
# ============================================================
def _get(d, *keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _team_name(team_obj):
    return _get(team_obj, "teamName", "TeamName", default="Unbekannt")


def get_matches_for_season(season: int) -> list[dict]:
    url = f"{BASE_URL_LIGA}/getmatchdata/{LEAGUE_SHORTCUT}/{season}"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return resp.json()


def get_finished_matches(seasons: list[int]) -> list[dict]:
    all_matches = []
    for season in seasons:
        for m in get_matches_for_season(season):
            if not _get(m, "matchIsFinished", "MatchIsFinished", default=False):
                continue
            results = _get(m, "matchResults", "MatchResults", default=[])
            final = next(
                (r for r in results if _get(r, "resultTypeID", "ResultTypeID") == 2),
                None,
            )
            if not final:
                continue
            team1 = _get(m, "team1", "Team1", default={})
            team2 = _get(m, "team2", "Team2", default={})
            hg = _get(final, "pointsTeam1", "PointsTeam1")
            ag = _get(final, "pointsTeam2", "PointsTeam2")
            if hg is None or ag is None:
                continue
            all_matches.append({
                "date": _get(m, "matchDateTimeUTC", "MatchDateTimeUTC"),
                "home": _team_name(team1),
                "away": _team_name(team2),
                "home_goals": int(hg),
                "away_goals": int(ag),
            })
    return all_matches


def get_upcoming_matches() -> list[dict]:
    """
    Naechster noch nicht gespielter Spieltag der aktuellen Saison.

    Nutzt bewusst NICHT den "ohne Saison"-Endpunkt von OpenLigaDB direkt -
    der zeigt zwischen zwei Spieltagen (z.B. nach Spieltag 1, bevor
    Spieltag 2 beginnt) manchmal noch den GERADE ABGESCHLOSSENEN Spieltag
    statt des naechsten. Dann waere "upcoming" faelschlich leer, tipps/ und
    docs/ wuerden nie angelegt, und der Commit-Schritt schlaegt fehl
    ("pathspec did not match any files"). Stattdessen: ganze Saison laden,
    selbst den naechsten Spieltag mit unfertigen Spielen bestimmen.
    """
    matches = get_matches_for_season(_aktuelle_saison)
    unfinished = [
        m for m in matches
        if not _get(m, "matchIsFinished", "MatchIsFinished", default=False)
    ]
    if not unfinished:
        return []

    def spieltag_of(m):
        gruppe = _get(m, "group", "Group", default={})
        return _get(gruppe, "groupOrderID", "GroupOrderID", default=999)

    naechster_spieltag = min(spieltag_of(m) for m in unfinished)
    naechste_spiele = [m for m in unfinished if spieltag_of(m) == naechster_spieltag]

    result = []
    for m in naechste_spiele:
        team1 = _get(m, "team1", "Team1", default={})
        team2 = _get(m, "team2", "Team2", default={})
        result.append({
            "match_id": _get(m, "matchID", "MatchID"),
            "date": _get(m, "matchDateTimeUTC", "MatchDateTimeUTC"),
            "home": _team_name(team1),
            "away": _team_name(team2),
        })
    return result


# ============================================================
# DIXON-COLES MODELL (Poisson oder Negative Binomial)
# ============================================================
def _tau(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _pmf(k, mean, dispersion):
    """dispersion=None -> Poisson. dispersion=r (endlich) -> Negative Binomial."""
    mean = max(mean, 1e-9)
    if dispersion is None:
        return poisson.pmf(k, mean)
    r = dispersion
    p = r / (r + mean)
    return nbinom.pmf(k, r, p)


def fit_model(matches, half_life_days=400, use_negative_binomial=USE_NEGATIVE_BINOMIAL):
    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    if n < 2:
        raise ValueError("Zu wenig Teams in den Trainingsdaten.")

    now = datetime.now(timezone.utc)
    weights = []
    for m in matches:
        d = datetime.fromisoformat(m["date"].replace("Z", "+00:00"))
        days_ago = max((now - d).days, 0)
        weights.append(0.5 ** (days_ago / half_life_days))

    n_base = 2 * n + 2

    def unpack(p):
        attack, defense, home_adv, rho = p[:n], p[n:2 * n], p[2 * n], p[2 * n + 1]
        r = np.exp(np.clip(p[n_base], -20, 20)) if use_negative_binomial else None
        return attack, defense, home_adv, rho, r

    def neg_log_likelihood(p):
        attack, defense, home_adv, rho, r = unpack(p)
        ll = 0.0
        for w, m in zip(weights, matches):
            i, j = idx[m["home"]], idx[m["away"]]
            lam = np.exp(attack[i] + defense[j] + home_adv)
            mu = np.exp(attack[j] + defense[i])
            x, y = m["home_goals"], m["away_goals"]
            p_xy = _pmf(x, lam, r) * _pmf(y, mu, r) * _tau(x, y, lam, mu, rho)
            ll += w * np.log(max(p_xy, 1e-10))
        return -ll

    n_params = n_base + (1 if use_negative_binomial else 0)
    x0 = np.zeros(n_params)
    x0[2 * n] = 0.3
    if use_negative_binomial:
        x0[n_base] = np.log(10.0)

    constraints = [{"type": "eq", "fun": lambda p: np.sum(p[:n])}]
    res = minimize(neg_log_likelihood, x0, constraints=constraints,
                    method="SLSQP", options={"maxiter": 400})

    attack, defense, home_adv, rho, r = unpack(res.x)
    return {
        "teams": teams,
        "attack": dict(zip(teams, attack)),
        "defense": dict(zip(teams, defense)),
        "home_advantage": float(home_adv),
        "rho": float(rho),
        "dispersion": float(r) if r is not None else None,
        "use_negative_binomial": use_negative_binomial,
        "converged": bool(res.success),
    }


def predict_score_matrix(model, home_team, away_team, max_goals=MAX_GOALS):
    attack, defense = model["attack"], model["defense"]
    home_adv, rho = model["home_advantage"], model["rho"]
    dispersion = model.get("dispersion")

    a_home = attack.get(home_team, AUFSTEIGER_ATTACK_DEFAULT)
    d_home = defense.get(home_team, AUFSTEIGER_DEFENSE_DEFAULT)
    a_away = attack.get(away_team, AUFSTEIGER_ATTACK_DEFAULT)
    d_away = defense.get(away_team, AUFSTEIGER_DEFENSE_DEFAULT)
    unsichere_daten = home_team not in attack or away_team not in attack

    lam = np.exp(a_home + d_away + home_adv)
    mu = np.exp(a_away + d_home)

    matrix = np.zeros((max_goals + 1, max_goals + 1))
    for x in range(max_goals + 1):
        for y in range(max_goals + 1):
            matrix[x, y] = max(
                _pmf(x, lam, dispersion) * _pmf(y, mu, dispersion) * _tau(x, y, lam, mu, rho), 0,
            )
    matrix /= matrix.sum()
    return matrix, unsichere_daten


# ============================================================
# KOPF-AN-KOPF-HISTORIE
# ============================================================
def compute_h2h_probs(matches, home_team, away_team,
                       min_duelle=H2H_MIN_DUELLE, max_duelle=H2H_MAX_DUELLE):
    relevante = [m for m in matches if {m["home"], m["away"]} == {home_team, away_team}]
    if len(relevante) < min_duelle:
        return None, len(relevante)
    relevante = sorted(relevante, key=lambda m: m["date"])[-max_duelle:]
    sieg_home = sieg_away = remis = 0
    for m in relevante:
        diff = (m["home_goals"] - m["away_goals"]) if m["home"] == home_team \
            else (m["away_goals"] - m["home_goals"])
        if diff > 0:
            sieg_home += 1
        elif diff < 0:
            sieg_away += 1
        else:
            remis += 1
    n = len(relevante)
    return {"home": sieg_home / n, "draw": remis / n, "away": sieg_away / n}, n


def h2h_gewicht(n_duelle, max_gewicht=0.35, pro_duell=0.08):
    return min(max_gewicht, pro_duell * n_duelle)


# ============================================================
# GENERISCHES BLENDING (fuer H2H und Marktquoten gleichermassen nutzbar)
# ============================================================
def _tendenz_wahrscheinlichkeiten(matrix):
    n = matrix.shape[0]
    idx = np.arange(n)
    return {
        "home": matrix[idx[:, None] > idx[None, :]].sum(),
        "draw": matrix[idx[:, None] == idx[None, :]].sum(),
        "away": matrix[idx[:, None] < idx[None, :]].sum(),
    }


def blend_probabilities(matrix, target_probs, weight):
    if not target_probs or weight <= 0:
        return matrix
    n = matrix.shape[0]
    idx = np.arange(n)
    masks = {
        "home": idx[:, None] > idx[None, :],
        "draw": idx[:, None] == idx[None, :],
        "away": idx[:, None] < idx[None, :],
    }
    model_probs = _tendenz_wahrscheinlichkeiten(matrix)
    out = matrix.copy()
    for key, mask in masks.items():
        model_p = model_probs[key]
        target_p = (1 - weight) * model_p + weight * target_probs[key]
        if model_p > 1e-9:
            out[mask] *= target_p / model_p
    out /= out.sum()
    return out


# ============================================================
# KICKTIPP-PUNKTOPTIMIERUNG (echte Regel: Sieg 2/3/5, Remis 2/-/5)
# ============================================================
def punkte(tipp, tatsaechlich):
    th, ta = tipp
    ah, aa = tatsaechlich
    if (th, ta) == (ah, aa):
        return PUNKTE_ERGEBNIS
    tendenz_tipp = np.sign(th - ta)
    tendenz_echt = np.sign(ah - aa)
    if tendenz_tipp != tendenz_echt:
        return 0
    if tendenz_echt == 0:
        return PUNKTE_TENDENZ
    if (th - ta) == (ah - aa):
        return PUNKTE_TORDIFFERENZ
    return PUNKTE_TENDENZ


def optimalen_tipp_waehlen(matrix, max_goals=MAX_GOALS):
    best_tipp, best_ev = (0, 0), -1.0
    for th in range(max_goals + 1):
        for ta in range(max_goals + 1):
            ev = 0.0
            for ah in range(max_goals + 1):
                row = matrix[ah]
                for aa in range(max_goals + 1):
                    p = row[aa]
                    if p > 0:
                        ev += p * punkte((th, ta), (ah, aa))
            if ev > best_ev:
                best_ev = ev
                best_tipp = (th, ta)
    return best_tipp, best_ev


# ============================================================
# MARKTQUOTEN (the-odds-api.com, optional - ODDS_API_KEY setzen)
# ============================================================
_IGNORE_TOKENS = {
    "1", "04", "05", "07", "1899", "fc", "sc", "sv", "sg", "tsg", "vfb",
    "vfl", "borussia", "verein", "fuer", "e", "v", "spvgg",
}


def _normalize(name):
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    name = name.lower()
    tokens = re.split(r"[^a-z0-9]+", name)
    return " ".join(t for t in tokens if t and t not in _IGNORE_TOKENS)


def _similarity(a, b):
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    if na in nb or nb in na:
        return 1.0
    return SequenceMatcher(None, na, nb).ratio()


def _best_match(target, candidates, threshold=0.6):
    best, best_score = None, 0.0
    for c in candidates:
        s = _similarity(target, c)
        if s > best_score:
            best, best_score = c, s
    return best if best_score >= threshold else None


def get_market_probabilities():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        print("ODDS_API_KEY nicht gesetzt - rechne ohne Quoten-Abgleich weiter.")
        return {}
    try:
        resp = requests.get(
            f"{BASE_URL_ODDS}/sports/{SPORT_KEY_ODDS}/odds",
            params={"apiKey": api_key, "regions": "eu", "markets": "h2h",
                    "oddsFormat": "decimal"},
            timeout=20,
        )
        resp.raise_for_status()
        events = resp.json()
    except requests.RequestException as e:
        print(f"Quoten-Abruf fehlgeschlagen ({e}) - rechne ohne Quoten-Abgleich weiter.")
        return {}

    result = {}
    for event in events:
        home_api, away_api = event.get("home_team"), event.get("away_team")
        if not home_api or not away_api:
            continue
        home_probs, draw_probs, away_probs = [], [], []
        for bm in event.get("bookmakers", []):
            h2h = next((m for m in bm.get("markets", []) if m.get("key") == "h2h"), None)
            if not h2h or len(h2h.get("outcomes", [])) != 3:
                continue
            raw = {}
            for o in h2h["outcomes"]:
                name, price = o.get("name"), o.get("price")
                if not price or price <= 1.0:
                    continue
                if name == home_api:
                    raw["home"] = 1 / price
                elif name == away_api:
                    raw["away"] = 1 / price
                else:
                    raw["draw"] = 1 / price
            if len(raw) != 3:
                continue
            total = sum(raw.values())
            home_probs.append(raw["home"] / total)
            draw_probs.append(raw["draw"] / total)
            away_probs.append(raw["away"] / total)
        if not home_probs:
            continue
        result[(home_api, away_api)] = {
            "home": sum(home_probs) / len(home_probs),
            "draw": sum(draw_probs) / len(draw_probs),
            "away": sum(away_probs) / len(away_probs),
        }
    return result


def match_probabilities_for_fixture(home_ol, away_ol, market_data):
    if not market_data:
        return None
    api_teams = {t for pair in market_data for t in pair}
    home_match = _best_match(home_ol, list(api_teams))
    away_match = _best_match(away_ol, list(api_teams))
    if not home_match or not away_match:
        return None
    return market_data.get((home_match, away_match))


# ============================================================
# HOMESCREEN-SEITE (docs/index.html, fuer GitHub Pages)
# ============================================================
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Kicktipp">
<title>Kicktipp-Tipps</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; padding: 20px 16px 40px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f1115; color: #f2f2f2; }}
  h1 {{ font-size: 1.25em; margin: 0 0 2px; }}
  .stand {{ color: #8a8f98; font-size: 0.82em; margin-bottom: 18px; }}
  .spiel {{ background: #1b1e26; border-radius: 14px; padding: 14px 16px; margin-bottom: 10px; border: 1px solid #262a33; }}
  .teams {{ font-size: 1.02em; display: flex; justify-content: space-between; gap: 8px; }}
  .heim, .ausw {{ flex: 1; }}
  .ausw {{ text-align: right; }}
  .tipp {{ font-weight: 700; color: #4ade80; font-size: 1.15em; padding: 0 10px; white-space: nowrap; }}
  .meta {{ font-size: 0.78em; color: #9aa0a8; margin-top: 6px; }}
  .badge {{ display: inline-block; padding: 1px 7px; border-radius: 20px; background: #262a33; margin-left: 6px; font-size: 0.75em; }}
  .badge.quoten {{ background: #16321f; color: #4ade80; }}
  .badge.h2h {{ background: #1e2a3a; color: #7dd3fc; }}
  footer {{ color: #565c66; font-size: 0.75em; margin-top: 24px; text-align: center; }}
</style>
</head>
<body>
<h1>Kicktipp-Vorschlaege</h1>
<div class="stand">Stand: {stand}</div>
{spiele}
<footer>Automatisch berechnet - Dixon-Coles-Modell{modell_hinweis}. Kein Ergebnis ist garantiert.</footer>
</body>
</html>
"""

_SPIEL_TEMPLATE = """<div class="spiel">
  <div class="teams">
    <span class="heim">{home}</span>
    <span class="tipp">{th}:{ta}</span>
    <span class="ausw">{away}</span>
  </div>
  <div class="meta">erw. Punkte: {ev:.2f}{badges}</div>
</div>
"""


def render_html(ergebnisse, model):
    spiele_html = []
    for e in ergebnisse:
        badges = ""
        if e.get("unsicher"):
            badges += '<span class="badge">wenig Daten</span>'
        if e.get("hat_h2h"):
            badges += '<span class="badge h2h">+H2H</span>'
        badges += '<span class="badge quoten">+Quoten</span>' if e.get("hat_quote") \
            else '<span class="badge">nur Modell</span>'
        spiele_html.append(_SPIEL_TEMPLATE.format(
            home=e["home"], away=e["away"], th=e["tipp"][0], ta=e["tipp"][1],
            ev=e["ev"], badges=badges,
        ))
    modell_name = "Negative Binomial" if model.get("use_negative_binomial") else "Poisson"
    return _HTML_TEMPLATE.format(
        stand=datetime.now().strftime("%d.%m.%Y %H:%M"),
        spiele="".join(spiele_html),
        modell_hinweis=f" ({modell_name})",
    )


# ============================================================
# E-MAIL
# ============================================================
def versende_email(text):
    absender = os.environ.get("SMTP_USER")
    passwort = os.environ.get("SMTP_PASSWORD")
    empfaenger = os.environ.get("EMPFAENGER_EMAIL", absender)
    if not absender or not passwort:
        print("SMTP_USER/SMTP_PASSWORD nicht gesetzt - E-Mail-Versand uebersprungen.")
        return
    msg = MIMEText(text)
    msg["Subject"] = "Deine Kicktipp-Vorschlaege"
    msg["From"] = absender
    msg["To"] = empfaenger
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(absender, passwort)
        server.send_message(msg)
    print(f"E-Mail an {empfaenger} verschickt.")


# ============================================================
# HAUPTABLAUF
# ============================================================
def main():
    print(f"Lade historische Ergebnisse (Saisons {TRAININGS_SAISONS}) ...")
    matches = get_finished_matches(TRAININGS_SAISONS)
    print(f"{len(matches)} abgeschlossene Spiele geladen.")
    if len(matches) < 20:
        print("Warnung: sehr wenig Trainingsdaten - Tipps werden ungenau sein.")

    print(f"Trainiere Modell (Negative Binomial={USE_NEGATIVE_BINOMIAL}) ...")
    model = fit_model(matches)
    disp_str = f", Dispersion r={model['dispersion']:.1f}" if model["dispersion"] else ""
    print(f"Konvergiert: {model['converged']}, Heimvorteil={model['home_advantage']:.2f}{disp_str}")

    print("Lade naechste Spiele ...")
    upcoming = get_upcoming_matches()
    if not upcoming:
        print("Keine anstehenden Spiele gefunden (evtl. Saisonpause/Interlaenderspiele).")
        return

    print("Hole Marktquoten (falls ODDS_API_KEY gesetzt) ...")
    market_data = get_market_probabilities()
    print(f"{len(market_data)} Spiele mit Quoten gefunden.")

    ergebnisse = []
    for spiel in upcoming:
        home, away = spiel["home"], spiel["away"]
        matrix, unsicher = predict_score_matrix(model, home, away)

        h2h_probs, n_h2h = compute_h2h_probs(matches, home, away)
        if h2h_probs:
            matrix = blend_probabilities(matrix, h2h_probs, weight=h2h_gewicht(n_h2h))

        market_probs = match_probabilities_for_fixture(home, away, market_data)
        matrix = blend_probabilities(matrix, market_probs, weight=ODDS_WEIGHT)

        tipp, ev = optimalen_tipp_waehlen(matrix)
        ergebnisse.append({
            "home": home, "away": away, "tipp": tipp, "ev": ev,
            "unsicher": unsicher, "hat_quote": market_probs is not None,
            "hat_h2h": h2h_probs is not None,
        })

    zeilen = [f"Kicktipp-Vorschlaege - Stand {datetime.now().strftime('%d.%m.%Y %H:%M')}", ""]
    for e in ergebnisse:
        hinweise = []
        if e["unsicher"]:
            hinweise.append("wenig Daten")
        if e["hat_h2h"]:
            hinweise.append("+H2H")
        hinweise.append("+Quoten" if e["hat_quote"] else "nur Modell")
        zeile = (f"{e['home']} {e['tipp'][0]}:{e['tipp'][1]} {e['away']}   "
                 f"[erw. Punkte: {e['ev']:.2f}]  ({', '.join(hinweise)})")
        zeilen.append(zeile)
        print(zeile)
    text = "\n".join(zeilen)

    os.makedirs(TIPPS_DIR, exist_ok=True)
    dateiname = os.path.join(TIPPS_DIR, f"{datetime.now().strftime('%Y-%m-%d_%H%M')}.txt")
    with open(dateiname, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Log geschrieben: {dateiname}")

    os.makedirs(DOCS_DIR, exist_ok=True)
    html = render_html(ergebnisse, model)
    with open(os.path.join(DOCS_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("Homescreen-Seite aktualisiert: docs/index.html")

    versende_email(text)


if __name__ == "__main__":
    main()
