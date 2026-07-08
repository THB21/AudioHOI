# Energy / Loss Analysis — Vortrags-Summary

Vollständige Analyse der Optimierungs-Energie (Loss-Terme) der AudioHOI-Objektrekonstruktion,
für den Vortrag aufbereitet. Drei Beispiele: **basketball, football, mug**.

Begleit-Artefakte (pro Video unter `results/`):
- **Loss-Kurve**: `diagnostics/energy/energy_loss_curve.png` (E_total + Komponenten, Teampartner-Format)
- **Video**: `renders/energy_video.mp4` (Szene oben, Live-Loss-Kurve mit Zeit-Cursor unten)
- **Pro-Term-Plots**: `diagnostics/energy/energy_terms.png`, **3D-Marker**: `trajectory_3d.png`
- **CSV** (wie Teampartner): `diagnostics/energy/our_loss_residuals.csv`

---

## 1. Was ist die Energie? (1 Folie)

Wir optimieren die Objekt-Pose `T_t` gegen den fixen, metrischen Menschen (GVHMR + HaMeR).
Pro Frame minimiert der Solver eine gewichtete Summe geometrischer Residuen:

```
E_total = w_2d·R_center + w_depth·R_depth + w_contact·R_contact
        + w_support·R_support + w_smooth·R_reg
```

**Kein gelerntes Prior, keine Physik-Annahme (keine `z=f·R/r` Kugelgröße).** Jeder Term ist
geometrisch/beobachtungsbasiert. Ein Term ≈ 0 = Constraint exakt erfüllt; ein großer Term =
Spannung oder unzuverlässige Quelle.

| Term | misst | Einheit |
|---|---|---|
| **R_center** (E_2d) | projiziertes 3D-Zentrum vs. 2D-Beobachtung | px |
| **R_depth** (E_depth) | \|gelöstes tz − DA3-Metrik-Tiefe\| | m |
| **R_contact** (E_contact) | 3D-Abstand Objekt ↔ Kontakt-Körperteil | m |
| **R_support** (E_support) | Objektboden vs. Bodenlinie an Bodenkontakten | px |
| **R_reg** (E_smooth) | Translations-Beschleunigung (Jerk) | m/frame² |

---

## 2. Ergebnisse — wer trägt die Energie? (1 Folie)

Anteil am `E_total` (Mittelwert):

| Term | basketball | football | mug |
|---|---|---|---|
| E_total (mean) | 0.31 | **1.76** | 0.07 |
| **E_depth** | 0.25 (**79 %**) | **1.46 (83 %)** | 0.05 (**78 %**) |
| E_smooth | 0.05 (15 %) | 0.04 (3 %) | 0.002 (3 %) |
| E_support | 0.02 (6 %) | 0.26 (15 %) | — |
| E_2d / E_contact | **0 / 0** | **0 / 0** | — |

---

## 3. Die drei Kernaussagen (Haupt-Folien)

### (A) 2D-Zentrum und Kontakt sind harte Anker → Residuum 0
`E_2d` und `E_contact` sind **überall exakt 0** — nicht zufällig, sondern weil der Solver sie
**exakt erzwingt**: das Objekt reprojiziert immer aufs beobachtete 2D-Zentrum, und an
Kontaktframes wird es ans Körperteil gepinnt. → **2D-Tracking + Kontakt sind das verlässliche
Rückgrat** jeder Lösung. (Hinweis: weil Kontakt ein *harter* Anker ist, erscheint er als 0; der
*rohe* Kontaktabstand ist die eigentlich interessante Größe — siehe (C).)

### (B) Tiefe dominiert die Energie überall (78–83 %) → schwächstes Glied
Monokulare metrische Tiefe ist der einzige lose Term. Seine **Größe = Tiefen-Zuverlässigkeit**:
- basketball **0.25 m** — brauchbarer weicher Cue (corr +0.69, conf 0.52)
- mug **0.05 m** — klein, aber Objekt bewegt sich kaum in Tiefe (corr +0.54)
- football **1.46 m** — riesig.

**Warum football so groß ist (und warum das korrekt ist):** DA3-Tiefe versagt dort
(corr +0.34, **conf 0.07**, Objekt extrapoliert 3× über den Körper-Bereich). Nach dem
Conf-Gating **ignoriert** der Solver diese schlechte Tiefe → tz weicht von DA3 ab → großes
E_depth. Das ist das System, das ehrlich sagt *„diesem Tiefen-Cue traue ich nicht"*, kein
Positionierungsfehler. Merksatz:
- großes E_depth **+ niedrige conf** = Tiefe wird (zu Recht) ignoriert (football) ✓
- großes E_depth **+ hohe conf** = echter Konflikt (tritt hier nicht auf)

### (C) Kontakt als volle 3D-Beschränkung (neu)
Vorher pinnte der Kontakt nur die **Tiefe (z)** → der Ball saß seitlich neben dem Fuß
(football: **1.1 m** weg, max 4.4 m, nie berührend). Neu: **VLM/LLM wählt das Kontakt-Teil**
(football→Fuß, basketball→Hand) → **volle 3D-Beschränkung** zieht das Objekt an Audio+Video-
Kontakten auf das Teil:

| | Objekt→Teil an Kontakten | berührend |
|---|---|---|
| football → Fuß | **1.107 → 0.158 m** | 0 → 12/15 |
| basketball → Hand | 0.153 → 0.120 m | 24 → 32/39 |

Artefakte: `pose6d_contact3d/contact_gap_before_after.png`, korrigiertes
`renders/full_scene_3d/world.mp4` (Ball sitzt am Fuß).

---

## 4. Vergleich zum Teampartner (optional, 1 Folie)

Gleiches Format (`stage7_loss_residuals.csv` ↔ unser `our_loss_residuals.csv`):
- **Unsere** Energie ist **tiefen-/kontakt-getrieben** (E_depth dominiert).
- Die des Teampartners ist **`E_prior`/`E_reg`-dominiert** (Pose-Prior; football-Spike bis 38.5).
→ Unterschiedliche Philosophie: wir vertrauen Beobachtungen (2D/Kontakt/Tiefe), das Prior trägt
bei uns kein Gewicht.

---

## 5. Take-home (Schluss-Folie)

1. Die verlässliche Geometrie (2D-Zentrum, Kontakt) wird **exakt** erfüllt → Residuum 0.
2. **Die gesamte Rest-Energie ist im Wesentlichen der Tiefen-Term**; seine Größe ist ein direkter
   Indikator, wie brauchbar monokulare Tiefe pro Video ist (klein: basketball/mug, groß und
   korrekt-ignoriert: football).
3. **Kontakt ist jetzt eine volle 3D-Beschränkung** mit VLM-gewähltem Körperteil → das Objekt
   sitzt am Kontaktframe tatsächlich am Fuß/an der Hand.

## Figuren-/Video-Index für die Folien
| Folie | Datei (pro Video) |
|---|---|
| Energie-Konzept | `method_losses.md` (Formel) |
| Loss-Kurve | `diagnostics/energy/energy_loss_curve.png` |
| Energie-Video | `renders/energy_video.mp4` |
| Pro-Term | `diagnostics/energy/energy_terms.png` |
| Tiefe-Ursache | `diagnostics/DA3_depth_failure_analysis.md` |
| Kontakt 3D | `pose6d_contact3d/contact_gap_before_after.png` + `renders/full_scene_3d/world.mp4` |
