"""
backtest.py - Eigenstaendiges Analyse-Skript, NICHT Teil des woechentlichen
Bots (main.py). Einmal ausfuehren: `python backtest.py`. Braucht echten
Internetzugang zu OpenLigaDB - laeuft deshalb bei dir, nicht in Claudes
Sandbox (dort ist Netzwerk gesperrt, konnte hier nur strukturell mit
erfundenen Daten getestet werden, s. Kommentare unten).

WAS ES MACHT (echter Walk-Forward-Test, keine Ruecksicht-Verzerrung):
Fuer jeden Spieltag ab START_AB_SPIELTAG wird das Modell NUR auf Spiele VOR
diesem Spieltag gefittet, sagt dann genau diesen Spieltag vorher, vergleicht
mit dem tatsaechlichen Ergebnis und summiert die echten Kicktipp-Punkte auf -
fuer mehrere Modell-Varianten (Poisson/Negative Binomial, mit/ohne
Kopf-an-Kopf-Gewichtung). Am Ende: eine Rangliste, welche Variante
tatsaechlich am meisten Punkte gebracht haette. Marktquoten sind bewusst
NICHT im Backtest (historische Quoten sind ueber die kostenlose API-Stufe
nicht ohne Weiteres verfuegbar) - das bleibt ein Punkt, der sich nur live
beobachten laesst.

Laufzeit: mehrere Minuten, da pro Spieltag und Konfiguration neu gefittet
wird (methodisch notwendig, sonst waere es keine ehrliche Vorhersage,
sondern Rueckschau mit Wissen aus der Zukunft).
"""
from main import (
    get_matches_for_season, fit_model, predict_score_matrix,
    optimalen_tipp_waehlen, punkte, compute_h2h_probs, blend_probabilities,
    h2h_gewicht, _get,
)

SAISONS = [2023, 2024, 2025]      # anpassen: mehr/andere Saisons testen
START_AB_SPIELTAG = 9             # vorher zu wenig Historie fuer einen fairen Fit
MIN_TRAININGSSPIELE = 50


def lade_alle_spiele_mit_spieltag(saisons):
    """Wie main.get_finished_matches(), behaelt aber Saison+Spieltag fuer
    die chronologische Walk-Forward-Sortierung."""
    alle = []
    for saison in saisons:
        for m in get_matches_for_season(saison):
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
            gruppe = _get(m, "group", "Group", default={})
            spieltag = _get(gruppe, "groupOrderID", "GroupOrderID", default=0)
            alle.append({
                "date": _get(m, "matchDateTimeUTC", "MatchDateTimeUTC"),
                "home": _get(team1, "teamName", "TeamName", default="?"),
                "away": _get(team2, "teamName", "TeamName", default="?"),
                "home_goals": int(hg),
                "away_goals": int(ag),
                "saison": saison,
                "spieltag": int(spieltag) if spieltag else 0,
            })
    return alle


def backtest(alle_spiele, use_negative_binomial, use_h2h):
    """
    Gibt (gesamt_punkte, anzahl_spiele, details) zurueck.
    details: Liste von (saison, spieltag, home, away, tipp, echt, punkte) -
    fuer alle, die genauer reinschauen wollen, wo es hakt.
    """
    eindeutige_spieltage = sorted({(m["saison"], m["spieltag"]) for m in alle_spiele})

    gesamt_punkte = 0
    n_spiele = 0
    details = []

    for saison, spieltag in eindeutige_spieltage:
        if spieltag < START_AB_SPIELTAG and saison == eindeutige_spieltage[0][0]:
            continue  # erste Saison: erst ab hier genug eigene Historie

        training = [
            m for m in alle_spiele
            if (m["saison"], m["spieltag"]) < (saison, spieltag)
        ]
        test = [
            m for m in alle_spiele
            if m["saison"] == saison and m["spieltag"] == spieltag
        ]
        if len(training) < MIN_TRAININGSSPIELE or not test:
            continue

        try:
            model = fit_model(training, use_negative_binomial=use_negative_binomial)
        except Exception as e:
            print(f"  Fit fehlgeschlagen ({saison}/{spieltag}): {e} - uebersprungen")
            continue

        for spiel in test:
            matrix, _ = predict_score_matrix(model, spiel["home"], spiel["away"])
            if use_h2h:
                h2h_probs, n_h2h = compute_h2h_probs(training, spiel["home"], spiel["away"])
                if h2h_probs:
                    matrix = blend_probabilities(matrix, h2h_probs, h2h_gewicht(n_h2h))
            tipp, _ = optimalen_tipp_waehlen(matrix)
            echt = (spiel["home_goals"], spiel["away_goals"])
            pkt = punkte(tipp, echt)
            gesamt_punkte += pkt
            n_spiele += 1
            details.append((saison, spieltag, spiel["home"], spiel["away"], tipp, echt, pkt))

    return gesamt_punkte, n_spiele, details


def main():
    print(f"Lade Spiele aus Saisons {SAISONS} von OpenLigaDB ...")
    alle_spiele = lade_alle_spiele_mit_spieltag(SAISONS)
    print(f"{len(alle_spiele)} abgeschlossene Spiele geladen.\n")

    konfigurationen = [
        ("Poisson, ohne H2H (= altes Modell v1)", False, False),
        ("Poisson, mit H2H", False, True),
        ("Negative Binomial, ohne H2H", True, False),
        ("Negative Binomial, mit H2H (= neues Modell v2)", True, True),
    ]

    ergebnisse = []
    for name, nb, h2h in konfigurationen:
        print(f"Teste: {name} ...")
        pkt, n, _ = backtest(alle_spiele, nb, h2h)
        avg = pkt / n if n else 0
        ergebnisse.append((name, pkt, n, avg))
        print(f"  -> {pkt} Punkte in {n} Spielen ({avg:.3f} Punkte/Spiel)\n")

    print("=" * 65)
    print("ERGEBNIS - sortiert nach Punkten/Spiel (hoeher = besser)")
    print("=" * 65)
    for name, pkt, n, avg in sorted(ergebnisse, key=lambda x: -x[3]):
        print(f"{name:48s} {avg:.3f} Pkt/Spiel  ({pkt} in {n})")

    beste = max(ergebnisse, key=lambda x: x[3])
    alte = next(e for e in ergebnisse if e[0].startswith("Poisson, ohne H2H"))
    diff = beste[3] - alte[3]
    print()
    if beste[0] != alte[0] and diff > 0.01:
        print(f"Empfehlung: '{beste[0]}' schlaegt das alte Modell v1 um "
              f"{diff:.3f} Punkte/Spiel im Backtest ({len(SAISONS)} Saisons, "
              f"{beste[2]} Vorhersagen). In main.py entsprechend setzen.")
    else:
        print("Kein Modell schlaegt v1 klar im Backtest - v1-Einstellungen "
              "beibehalten waere ebenfalls vertretbar.")


if __name__ == "__main__":
    main()
