"""
backtest.py - Eigenstaendiges Analyse-Skript, NICHT Teil des woechentlichen
Bots. Einmal ausfuehren (lokal oder ueber den "Kicktipp Backtest"-Workflow).
Braucht echten Internetzugang zu OpenLigaDB.

WAS ES MACHT (echter Walk-Forward-Test, keine Ruecksicht-Verzerrung):
Fuer jeden Spieltag ab START_AB_SPIELTAG wird das Modell NUR auf Spiele VOR
diesem Spieltag gefittet, sagt dann genau diesen Spieltag vorher, vergleicht
mit dem tatsaechlichen Ergebnis und summiert die echten Kicktipp-Punkte auf -
fuer vier Modell-Varianten (Poisson/Negative Binomial, mit/ohne
Kopf-an-Kopf-Gewichtung).

EFFIZIENZ-KORREKTUR (wichtig): Pro Spieltag wird nur EINMAL pro Torverteilung
gefittet (Poisson, Negative Binomial) - macht 2 Fits statt 4. H2H aendert
nichts am Fit selbst (das passiert erst danach, als Anpassung an die fertige
Vorhersage-Matrix), die "mit H2H"- und "ohne H2H"-Auswertung werden deshalb
aus demselben Fit abgeleitet statt ihn unnoetig doppelt zu berechnen. Die
erste Version dieses Skripts hat das nicht beachtet und haette dadurch (in
Kombination mit einer unvektorisierten, langsamen Fit-Funktion) mehrere
Stunden statt Minuten gebraucht.

Marktquoten sind bewusst NICHT im Backtest (historische Quoten sind ueber
die kostenlose API-Stufe nicht ohne Weiteres verfuegbar).
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


def backtest_alle_konfigurationen(alle_spiele):
    """
    Ein Durchlauf durch alle Spieltage, der ALLE VIER Konfigurationen
    gleichzeitig fuellt - fittet aber nur 2x pro Spieltag (Poisson, NB),
    nicht 4x. Gibt {(use_nb, use_h2h): {"punkte":, "n":}} zurueck.
    """
    eindeutige_spieltage = sorted({(m["saison"], m["spieltag"]) for m in alle_spiele})
    erste_saison = eindeutige_spieltage[0][0] if eindeutige_spieltage else None

    ergebnisse = {
        (False, False): {"punkte": 0, "n": 0},
        (False, True): {"punkte": 0, "n": 0},
        (True, False): {"punkte": 0, "n": 0},
        (True, True): {"punkte": 0, "n": 0},
    }

    anzahl_spieltage_verarbeitet = 0
    for saison, spieltag in eindeutige_spieltage:
        if saison == erste_saison and spieltag < START_AB_SPIELTAG:
            continue

        training = [m for m in alle_spiele if (m["saison"], m["spieltag"]) < (saison, spieltag)]
        test = [m for m in alle_spiele if m["saison"] == saison and m["spieltag"] == spieltag]
        if len(training) < MIN_TRAININGSSPIELE or not test:
            continue

        for use_nb in (False, True):
            try:
                model = fit_model(training, use_negative_binomial=use_nb)
            except Exception as e:
                print(f"  Fit fehlgeschlagen ({saison}/{spieltag}, NB={use_nb}): {e} - uebersprungen")
                continue

            for spiel in test:
                echt = (spiel["home_goals"], spiel["away_goals"])
                matrix, _ = predict_score_matrix(model, spiel["home"], spiel["away"])

                tipp_ohne, _ = optimalen_tipp_waehlen(matrix)
                ergebnisse[(use_nb, False)]["punkte"] += punkte(tipp_ohne, echt)
                ergebnisse[(use_nb, False)]["n"] += 1

                h2h_probs, n_h2h = compute_h2h_probs(training, spiel["home"], spiel["away"])
                matrix_h2h = matrix
                if h2h_probs:
                    matrix_h2h = blend_probabilities(matrix, h2h_probs, h2h_gewicht(n_h2h))
                tipp_mit, _ = optimalen_tipp_waehlen(matrix_h2h)
                ergebnisse[(use_nb, True)]["punkte"] += punkte(tipp_mit, echt)
                ergebnisse[(use_nb, True)]["n"] += 1

        anzahl_spieltage_verarbeitet += 1
        if anzahl_spieltage_verarbeitet % 10 == 0:
            print(f"  ... {anzahl_spieltage_verarbeitet} Spieltage verarbeitet "
                  f"(zuletzt Saison {saison}, Spieltag {spieltag})")

    return ergebnisse


def main():
    print(f"Lade Spiele aus Saisons {SAISONS} von OpenLigaDB ...")
    alle_spiele = lade_alle_spiele_mit_spieltag(SAISONS)
    print(f"{len(alle_spiele)} abgeschlossene Spiele geladen.\n")

    print("Starte Backtest (2 Fits pro Spieltag, beide H2H-Varianten daraus abgeleitet) ...")
    ergebnisse = backtest_alle_konfigurationen(alle_spiele)

    namen = {
        (False, False): "Poisson, ohne H2H (= main.py v1)",
        (False, True): "Poisson, mit H2H",
        (True, False): "Negative Binomial, ohne H2H",
        (True, True): "Negative Binomial, mit H2H (= main.py v2/v3 aktuell)",
    }

    print()
    print("=" * 65)
    print("ERGEBNIS - sortiert nach Punkten/Spiel (hoeher = besser)")
    print("=" * 65)
    tabelle = []
    for key, name in namen.items():
        r = ergebnisse[key]
        avg = r["punkte"] / r["n"] if r["n"] else 0
        tabelle.append((name, r["punkte"], r["n"], avg))
    for name, pkt, n, avg in sorted(tabelle, key=lambda x: -x[3]):
        print(f"{name:48s} {avg:.3f} Pkt/Spiel  ({pkt} in {n})")

    beste = max(tabelle, key=lambda x: x[3])
    alte = next(t for t in tabelle if t[0].startswith("Poisson, ohne H2H"))
    diff = beste[3] - alte[3]
    print()
    if beste[0] != alte[0] and diff > 0.01:
        print(f"Empfehlung: '{beste[0]}' schlaegt das alte v1-Modell um "
              f"{diff:.3f} Punkte/Spiel im Backtest ueber {len(SAISONS)} Saisons "
              f"({beste[2]} Vorhersagen).")
    else:
        print("Kein Modell schlaegt v1 klar im Backtest - v1-Einstellungen "
              "beibehalten waere ebenfalls vertretbar.")


if __name__ == "__main__":
    main()
