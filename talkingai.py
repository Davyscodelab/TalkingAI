#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TalkingAI — twee lokale Ollama-modellen debatteren met elkaar.

Werking in het kort
-------------------
1. De configuratie (onderwerpen + identiteiten van partner A en B) wordt
   gelezen uit ``info.json``.
2. Voor elk gekozen debat:
   - wordt gecontroleerd of Ollama draait en beide modellen beschikbaar zijn;
   - antwoorden partner A en partner B om beurt, met streaming output;
   - houdt elk model zijn volledige gespreksgeschiedenis bij;
   - wordt elk bericht meteen weggeschreven naar een Markdown-log in ``logs/``;
   - wordt na afloop automatisch een samenvatting gemaakt.
3. Ctrl+C onderbreekt netjes: wat al gezegd is, blijft bewaard in de log.

Gebruik
-------
    python talkingai.py                     # alle debatten uit info.json
    python talkingai.py --lijst             # toon beschikbare debatten
    python talkingai.py --debat ai-werk     # enkel het debat met id 'ai-werk'
    python talkingai.py --rondes 2          # korte testrun (overschrijft config)

Vereisten: Python 3.10+, Ollama (lokaal draaiend), ``pip install ollama``.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from ollama import Client, ResponseError
except ImportError:  # pragma: no cover - enkel bij ontbrekende dependency
    print("❌ De Python-bibliotheek 'ollama' ontbreekt. Installeer met: pip install ollama")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constanten
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "info.json"

# Standaardwaarden; alles kan overschreven worden in info.json → "instellingen".
DEFAULT_SETTINGS: dict[str, Any] = {
    "model_a": "llama3.1",
    "model_b": "qwen3",
    "rondes": 50,                 # 1 ronde = A spreekt + B spreekt
    "taal": "Nederlands",
    "max_woorden": 150,           # richtlijn per beurt, houdt het debat vlot
    "temperatuur": 0.8,
    "num_ctx": 16384,             # contextvenster in tokens (zie README)
    "samenvatting": True,
    "samenvatting_model": None,   # None = model van partner A
    "samenvatting_blok": 10,      # rondes per deelsamenvatting (past in num_ctx)
    "logmap": "logs",
    "ollama_host": "http://localhost:11434",
    # --- Kwaliteit van het debat (tegen herhaling en "te snel akkoord") ---
    "rondeteller": True,          # model krijgt elke beurt "ronde x van N" + opdracht
    "herhaling_drempel": 0.45,    # 0–1: vanaf hoeveel gelijkenis een bericht "herhaling" is
    "herhaling_vergelijk": 4,     # met hoeveel vorige berichten vergelijken
    "herhaling_herkansing": True, # bij herhaling één nieuwe poging vragen
    "repeat_penalty": 1.15,       # Ollama: straf op herhaalde woorden (standaard 1.1)
    "repeat_last_n": 256,         # Ollama: hoe ver terug die straf kijkt (standaard 64)
    "weigering_herkansing": True, # bij een weigering ("ik kan dit niet…") één nieuwe poging
    "openingen_vermijden": 3,     # aantal eigen vorige beurten waarvan opening/afsluiter vermeden wordt (0 = uit)
}

# ANSI-kleuren voor de console (werkt in Windows Terminal en recente cmd).
KLEUR_A = "\033[96m"      # cyaan
KLEUR_B = "\033[93m"      # geel
KLEUR_INFO = "\033[90m"   # grijs
KLEUR_FOUT = "\033[91m"   # rood
KLEUR_OK = "\033[92m"     # groen
VET = "\033[1m"
RESET = "\033[0m"

# Sommige modellen (bv. qwen3) kunnen "denk"-tekst tussen <think>-tags tonen.
THINK_TAG = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Eigen foutklassen
# ---------------------------------------------------------------------------

class ConfigFout(Exception):
    """Fout in info.json (ontbrekend, ongeldig of onvolledig)."""


class ModelFout(Exception):
    """Ollama draait niet of een model is niet beschikbaar."""


# ---------------------------------------------------------------------------
# Datastructuren
# ---------------------------------------------------------------------------

@dataclass
class Partner:
    """Eén deelnemer aan het debat."""

    label: str              # "A" of "B"
    naam: str               # weergavenaam in console en log
    identiteit: str         # wie/wat deze partner is (één woord of een alinea)
    standpunt: str          # wat deze partner verdedigt (mag leeg zijn)
    model: str              # volledige Ollama-modelnaam, bv. "llama3.1:8b"
    kleur: str
    kan_denken: bool = False                      # model ondersteunt 'think'
    geschiedenis: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Debat:
    """Eén debat uit info.json."""

    id: str
    titel: str
    onderwerp: str
    partner_a: dict[str, str]
    partner_b: dict[str, str]
    rondes: int


# ---------------------------------------------------------------------------
# Console-hulpfuncties
# ---------------------------------------------------------------------------

def console_voorbereiden() -> None:
    """Zorg voor UTF-8 output en ANSI-kleuren, ook op Windows."""
    if os.name == "nt":
        os.system("")  # activeert ANSI-escape codes in de Windows-console
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def info(tekst: str) -> None:
    print(f"{KLEUR_INFO}{tekst}{RESET}", flush=True)


def fout(tekst: str) -> None:
    print(f"{KLEUR_FOUT}❌ {tekst}{RESET}", file=sys.stderr, flush=True)


def nu() -> datetime:
    return datetime.now()


# ---------------------------------------------------------------------------
# Configuratie
# ---------------------------------------------------------------------------

def _verplichte_tekst(bron: dict[str, Any], sleutel: str, context: str) -> str:
    """Haal een niet-lege tekstwaarde op, of geef een duidelijke fout."""
    waarde = bron.get(sleutel)
    if not isinstance(waarde, str) or not waarde.strip():
        raise ConfigFout(f"{context}: '{sleutel}' ontbreekt of is leeg.")
    return waarde.strip()


def _partner_config(ruw: Any, standaardnaam: str, context: str) -> dict[str, str]:
    """
    Een partner mag als korte tekst ("Optimist") of als object
    ({"naam": "...", "identiteit": "...", "standpunt": "..."}) opgegeven worden.
    'standpunt' is optioneel, maar sterk aanbevolen: zonder vast standpunt
    gaan kleine modellen snel akkoord met elkaar.
    """
    if isinstance(ruw, str) and ruw.strip():
        tekst = ruw.strip()
        # Kort (bv. "Barista") → ook als naam gebruiken; een alinea niet.
        naam = tekst if len(tekst) <= 40 and "\n" not in tekst else standaardnaam
        return {"naam": naam, "identiteit": tekst, "standpunt": ""}
    if isinstance(ruw, dict):
        identiteit = _verplichte_tekst(ruw, "identiteit", context)
        naam = str(ruw.get("naam") or standaardnaam).strip()
        standpunt = str(ruw.get("standpunt") or "").strip()
        return {"naam": naam, "identiteit": identiteit, "standpunt": standpunt}
    raise ConfigFout(f"{context}: geef een tekst of een object met 'identiteit'.")


def config_laden(pad: Path) -> tuple[dict[str, Any], list[Debat]]:
    """Lees en valideer info.json. Geeft (instellingen, debatten) terug."""
    if not pad.is_file():
        raise ConfigFout(f"Configuratiebestand niet gevonden: {pad}")

    try:
        data = json.loads(pad.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ConfigFout(f"{pad.name} is geen geldige JSON (regel {e.lineno}, kolom {e.colno}): {e.msg}") from e

    if not isinstance(data, dict):
        raise ConfigFout(f"{pad.name} moet een JSON-object bevatten.")

    # Instellingen: standaardwaarden aangevuld met wat in de config staat.
    instellingen = {**DEFAULT_SETTINGS, **(data.get("instellingen") or {})}
    try:
        instellingen["rondes"] = int(instellingen["rondes"])
        instellingen["max_woorden"] = int(instellingen["max_woorden"])
        instellingen["num_ctx"] = int(instellingen["num_ctx"])
        instellingen["temperatuur"] = float(instellingen["temperatuur"])
        instellingen["samenvatting_blok"] = int(instellingen["samenvatting_blok"])
        instellingen["herhaling_drempel"] = float(instellingen["herhaling_drempel"])
        instellingen["herhaling_vergelijk"] = int(instellingen["herhaling_vergelijk"])
        instellingen["repeat_penalty"] = float(instellingen["repeat_penalty"])
        instellingen["repeat_last_n"] = int(instellingen["repeat_last_n"])
        instellingen["openingen_vermijden"] = int(instellingen["openingen_vermijden"])
    except (TypeError, ValueError) as e:
        raise ConfigFout(f"Ongeldige numerieke waarde in 'instellingen': {e}") from e
    if instellingen["rondes"] < 1:
        raise ConfigFout("'rondes' moet minstens 1 zijn.")
    if instellingen["samenvatting_blok"] < 1:
        raise ConfigFout("'samenvatting_blok' moet minstens 1 zijn.")

    ruwe_debatten = data.get("debatten")
    if not isinstance(ruwe_debatten, list) or not ruwe_debatten:
        raise ConfigFout("'debatten' ontbreekt of is leeg. Voeg minstens één debat toe.")

    debatten: list[Debat] = []
    gezien: set[str] = set()
    for i, d in enumerate(ruwe_debatten, start=1):
        ctx = f"debat #{i}"
        if not isinstance(d, dict):
            raise ConfigFout(f"{ctx}: moet een object zijn.")
        onderwerp = _verplichte_tekst(d, "onderwerp", ctx)
        debat_id = str(d.get("id") or f"debat-{i}").strip()
        if debat_id in gezien:
            raise ConfigFout(f"{ctx}: id '{debat_id}' komt meer dan één keer voor.")
        gezien.add(debat_id)
        titel = str(d.get("titel") or _korte_titel(onderwerp)).strip()
        debatten.append(Debat(
            id=debat_id,
            titel=titel,
            onderwerp=onderwerp,
            partner_a=_partner_config(d.get("partner_a"), "Partner A", f"{ctx} → partner_a"),
            partner_b=_partner_config(d.get("partner_b"), "Partner B", f"{ctx} → partner_b"),
            rondes=int(d.get("rondes") or instellingen["rondes"]),
        ))
    return instellingen, debatten


def _korte_titel(tekst: str, max_lengte: int = 70) -> str:
    """Maak een leesbare titel van (de eerste zin van) het onderwerp."""
    eerste = re.split(r"(?<=[.?!])\s", tekst.strip(), maxsplit=1)[0]
    return eerste if len(eerste) <= max_lengte else eerste[: max_lengte - 1].rstrip() + "…"


def _slug(tekst: str) -> str:
    """Bestandsnaamvriendelijke versie van een tekst (voor de logbestandsnaam)."""
    tekst = re.sub(r"[^\w\s-]", "", tekst, flags=re.UNICODE).strip().lower()
    return re.sub(r"[\s_-]+", "-", tekst)[:50] or "debat"


# ---------------------------------------------------------------------------
# Ollama: beschikbaarheid controleren
# ---------------------------------------------------------------------------

def geinstalleerde_modellen(client: Client) -> list[str]:
    """Lijst van lokaal beschikbare modellen, of ModelFout als Ollama niet draait."""
    try:
        antwoord = client.list()
    except ConnectionError as e:
        raise ModelFout(
            "Kan geen verbinding maken met Ollama. Draait Ollama? "
            "Start de Ollama-app of voer 'ollama serve' uit."
        ) from e
    return [m.model for m in antwoord.models if m.model]


def model_oplossen(gewenst: str, beschikbaar: list[str]) -> str:
    """
    Zoek de juiste lokale modelnaam.

    'llama3.1' matcht exact, of 'llama3.1:latest', of anders de eerste
    geïnstalleerde variant zoals 'llama3.1:8b'.
    """
    if gewenst in beschikbaar:
        return gewenst
    if ":" not in gewenst:
        if f"{gewenst}:latest" in beschikbaar:
            return f"{gewenst}:latest"
        varianten = sorted(m for m in beschikbaar if m.startswith(f"{gewenst}:"))
        if varianten:
            return varianten[0]
    lijst = "\n   - ".join(beschikbaar) or "(geen modellen gevonden)"
    raise ModelFout(
        f"Model '{gewenst}' is niet beschikbaar in Ollama.\n"
        f"   Installeer het met: ollama pull {gewenst}\n"
        f"   Beschikbare modellen:\n   - {lijst}"
    )


def kan_denken(client: Client, model: str) -> bool:
    """True als het model 'thinking' ondersteunt (dan zetten we dat uit)."""
    try:
        caps = client.show(model).capabilities or []
        return "thinking" in caps
    except Exception:  # show() is optioneel; bij twijfel niets meegeven
        return False


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def systeemprompt(ik: Partner, ander: Partner, debat: Debat, inst: dict[str, Any]) -> str:
    """Rolbeschrijving die elk model als eerste bericht krijgt."""
    # Een vast standpunt voorkomt dat beide modellen na enkele rondes
    # "overeenstemming bereiken" en het debat stilvalt.
    standpunt = ""
    if ik.standpunt:
        standpunt += f"JOUW STANDPUNT (verdedig dit het hele debat):\n{ik.standpunt}\n\n"
    if ander.standpunt:
        standpunt += f"STANDPUNT VAN {ander.naam.upper()}:\n{ander.standpunt}\n\n"
    return (
        f"Je neemt deel aan een debat van {debat.rondes} rondes.\n\n"
        f"JOUW IDENTITEIT ({ik.naam}):\n{ik.identiteit}\n\n"
        f"JE GESPREKSPARTNER ({ander.naam}):\n{ander.identiteit}\n\n"
        f"ONDERWERP / STELLING:\n{debat.onderwerp}\n\n"
        f"{standpunt}"
        f"REGELS:\n"
        f"- Spreek altijd in het {inst['taal']}.\n"
        f"- Blijf volledig in je rol en spreek vanuit je eigen identiteit.\n"
        f"- Blijf bij je eigen standpunt. Geef niet toe, zoek geen consensus en "
        f"zeg niet dat jullie het eens zijn.\n"
        f"- Reageer telkens kritisch op het laatste argument van {ander.naam}.\n"
        f"- Breng elke beurt een NIEUW argument, voorbeeld, cijfer of tegenvraag; "
        f"herhaal niets wat al gezegd is, ook niet in andere woorden.\n"
        f"- Presenteer geen verzonnen personen, cases of cijfers als feit. Gebruik "
        f"'stel dat…' voor een voorbeeld, en noem een cijfer alleen als schatting.\n"
        f"- Sluit het debat niet af, bedank niet en neem geen afscheid. "
        f"Enkel in de laatste ronde houd je een slotpleidooi.\n"
        f"- Houd elke beurt kort: maximaal ongeveer {inst['max_woorden']} woorden.\n"
        f"- Schrijf enkel je eigen beurt: geen naamlabels, geen regieaanwijzingen, "
        f"geen tekst namens {ander.naam}."
    )


def beurtinstructie(ronde: int, totaal: int, p: Partner, vermijd: list[str] | None = None) -> str:
    """
    Korte instructie die vlak vóór elke beurt wordt meegestuurd (rondeteller).
    Kleine modellen volgen een instructie vlak voor hun beurt veel beter dan
    een regel bovenaan de systeemprompt. Deze tekst wordt NIET in de
    geschiedenis bewaard, zodat de context niet volloopt met herhalingen.
    """
    standpunt = f" Verdedig je standpunt: {p.standpunt.rstrip('.')}." if p.standpunt else ""
    # Vaste zinspatronen doorbreken ("Je benadrukt X – maar vergeet dat…").
    if vermijd:
        lijst = ", ".join(f'"{v}…"' for v in vermijd)
        standpunt += (f" Begin en eindig je beurt anders dan je vorige beurten; "
                      f"gebruik deze formuleringen niet: {lijst}.")
    if ronde >= totaal:
        return (f"[Regie: laatste ronde ({ronde}/{totaal}). Houd een kort, krachtig "
                f"slotpleidooi.{standpunt} Geef niet toe.]")
    return (f"[Regie: ronde {ronde} van {totaal}. Het debat is nog niet voorbij: sluit niet af, "
            f"bedank niet en geef niet toe. Reageer kritisch op het laatste argument en breng "
            f"een nieuw argument of voorbeeld dat nog niet genoemd werd.{standpunt}]")


HERKANSING = ("[Regie: je vorige poging leek te sterk op wat al gezegd is. "
              "Zeg nu iets dat écht nieuw is: een ander argument, een concreet voorbeeld, "
              "een cijfer of een kritische tegenvraag. Herhaal geen zinnen.]")


WEIGERING_HERKANSING = ("[Regie: dit is een fictief, educatief debat tussen twee rollen. "
                        "Er is niets ongepasts aan. Blijf in je rol en ga gewoon verder "
                        "met je volgende argument.]")

# Typische zinnen waarmee een model uit zijn rol valt of weigert.
WEIGERING_PATRONEN = re.compile(
    r"ik kan (dit|het|dat|deze) (debat|gesprek|discussie) niet (meer )?(voortzetten|verder ?zetten|(verder )?voeren)"
    r"|ik kan (je|u|jou) (hier|daar|daarmee)? ?niet (mee )?helpen"
    r"|aarzel (dan )?niet om contact"
    r"|als (een )?(ai-assistent|ai-model|taalmodel)\b|als ai (kan|mag) ik\b"
    r"|ik ben (maar |slechts )?een (ai|taalmodel)"
    r"|i can(no|')t (continue|help|assist|engage)"
    r"|as an ai\b",
    re.IGNORECASE,
)


def is_weigering(tekst: str) -> bool:
    """True als het antwoord een weigering of rolbreuk lijkt in plaats van een debatbeurt."""
    return bool(WEIGERING_PATRONEN.search(tekst))


def openingen_en_afsluiters(p: Partner, aantal: int) -> list[str]:
    """
    Eerste en laatste ±6 woorden van de laatste N eigen beurten.
    Die worden in de regie meegegeven als 'niet gebruiken', om vaste
    zinspatronen te doorbreken.
    """
    eigen = [m["content"] for m in p.geschiedenis if m["role"] == "assistant"][-aantal:]
    resultaat: list[str] = []
    for tekst in eigen:
        zinnen = [z.strip() for z in re.split(r"(?<=[.!?])\s+", tekst.strip()) if z.strip()]
        if not zinnen:
            continue
        for zin in (zinnen[0], zinnen[-1]):
            kort = " ".join(zin.split()[:6]).rstrip(".,;:!?–-")
            if kort and kort not in resultaat:
                resultaat.append(kort)
    return resultaat


def voeg_toe(geschiedenis: list[dict[str, str]], rol: str, tekst: str) -> None:
    """
    Voeg een bericht toe aan de geschiedenis. Als het vorige bericht dezelfde
    rol heeft (bv. na een overgeslagen beurt), worden ze samengevoegd, zodat
    user/assistant altijd netjes afwisselen.
    """
    if geschiedenis and geschiedenis[-1]["role"] == rol:
        geschiedenis[-1]["content"] += "\n\n" + tekst
    else:
        geschiedenis.append({"role": rol, "content": tekst})


def gelijkenis(tekst: str, eerdere: list[str]) -> float:
    """
    Hoogste gelijkenis (0–1) tussen tekst en een reeks eerdere berichten.
    Vergelijkt op woordniveau (zonder hoofdletters en leestekens), zodat
    'bijna letterlijk kopiëren' gedetecteerd wordt.
    """
    def woorden(t: str) -> list[str]:
        return re.findall(r"\w+", t.lower())

    nieuw = woorden(tekst)
    if not nieuw:
        return 0.0
    return max((difflib.SequenceMatcher(None, nieuw, woorden(e), autojunk=False).ratio()
                for e in eerdere), default=0.0)


def openingsbericht(debat: Debat) -> str:
    """Eerste 'user'-bericht voor partner A, die het debat opent."""
    return (
        f"Open het debat. Geef je standpunt over het onderwerp en je sterkste "
        f"eerste argument.\n\nOnderwerp: {debat.onderwerp}"
    )


# ---------------------------------------------------------------------------
# Markdown-log
# ---------------------------------------------------------------------------

class MarkdownLog:
    """
    Schrijft het debat stap voor stap weg (append + flush).
    Daardoor blijft alles bewaard, ook bij een crash of Ctrl+C.
    """

    def __init__(self, map_: Path, debat: Debat, start: datetime) -> None:
        map_.mkdir(parents=True, exist_ok=True)
        naam = f"{start:%Y-%m-%d_%H-%M-%S}_{_slug(debat.id)}.md"
        self.pad = map_ / naam

    def schrijf(self, tekst: str) -> None:
        with self.pad.open("a", encoding="utf-8", newline="\n") as f:
            f.write(tekst)
            f.flush()

    def kop(self, debat: Debat, a: Partner, b: Partner, start: datetime) -> None:
        self.schrijf(
            f"# {debat.titel}\n\n"
            f"**Gestart:** {start:%Y-%m-%d %H:%M:%S}  \n"
            f"**Debat-id:** `{debat.id}`  \n"
            f"**Rondes (gepland):** {debat.rondes}\n\n"
            f"> [!warning] Entertainment, geen advies\n"
            f"> Dit debat is volledig gegenereerd door lokale taalmodellen. Cijfers, wetsartikelen, "
            f"voorbeelden en \"cases\" kunnen verzonnen of fout zijn. Gebruik niets hiervan als "
            f"financieel, juridisch of ander advies: lezen op eigen risico.\n\n"
            f"## Onderwerp\n\n{debat.onderwerp}\n\n"
            f"## Deelnemers\n\n"
            f"| | Naam | Model | Identiteit | Standpunt |\n"
            f"|---|---|---|---|---|\n"
            f"| A | {a.naam} | `{a.model}` | {_tabelcel(a.identiteit)} | {_tabelcel(a.standpunt) or '–'} |\n"
            f"| B | {b.naam} | `{b.model}` | {_tabelcel(b.identiteit)} | {_tabelcel(b.standpunt) or '–'} |\n\n"
            f"---\n\n## Gesprek\n\n"
        )

    def ronde(self, nummer: int) -> None:
        self.schrijf(f"### Ronde {nummer}\n\n")

    def bericht(self, p: Partner, tekst: str, tijd: datetime, onderbroken: bool = False,
                opmerking: str = "") -> None:
        extra = " _(onderbroken)_" if onderbroken else ""
        if opmerking:
            extra += f" _({opmerking})_"
        self.schrijf(f"**{p.label} — {p.naam}** · {tijd:%H:%M:%S}{extra}\n\n{tekst.strip()}\n\n")

    def samenvatting(self, model: str, tekst: str,
                     delen: list[tuple[str, str]] | None = None) -> None:
        """Eindsamenvatting, optioneel met inklapbare deelsamenvattingen (Obsidian-callout)."""
        blok = f"---\n\n## Samenvatting\n\n_Gemaakt door `{model}`_\n\n{tekst.strip()}\n\n"
        if delen:
            blok += "> [!note]- Deelsamenvattingen per blok\n"
            for titel, inhoud in delen:
                blok += f">\n> **{titel}**\n>\n"
                blok += "".join(f"> {regel}\n" if regel else ">\n" for regel in inhoud.strip().splitlines())
            blok += "\n"
        self.schrijf(blok)

    def afsluiting(self, status: str, berichten: int, rondes: int, start: datetime, einde: datetime,
                   herhalingen: int = 0, herkansingen: int = 0,
                   weigeringen: int = 0, overgeslagen: int = 0) -> None:
        duur = einde - start
        uren, rest = divmod(int(duur.total_seconds()), 3600)
        minuten, seconden = divmod(rest, 60)
        self.schrijf(
            f"---\n\n## Statistieken\n\n"
            f"| Gegeven | Waarde |\n|---|---|\n"
            f"| Status | {status} |\n"
            f"| Voltooide rondes | {rondes} |\n"
            f"| Uitgewisselde berichten | {berichten} |\n"
            f"| Herhalingen gedetecteerd | {herhalingen} |\n"
            f"| Herkansingen gevraagd | {herkansingen} |\n"
            f"| Weigeringen gedetecteerd | {weigeringen} |\n"
            f"| Beurten overgeslagen | {overgeslagen} |\n"
            f"| Gestart | {start:%Y-%m-%d %H:%M:%S} |\n"
            f"| Beëindigd | {einde:%Y-%m-%d %H:%M:%S} |\n"
            f"| Duur | {uren:02d}:{minuten:02d}:{seconden:02d} |\n"
        )


def _tabelcel(tekst: str) -> str:
    """Maak tekst veilig voor een Markdown-tabelcel."""
    return tekst.replace("|", "\\|").replace("\n", "<br>")


# ---------------------------------------------------------------------------
# Kern: één beurt met streaming
# ---------------------------------------------------------------------------

class OnderbrokenTijdensBeurt(Exception):
    """Ctrl+C tijdens het streamen; bevat de tekst die al ontvangen was."""

    def __init__(self, gedeeltelijk: str) -> None:
        super().__init__("onderbroken")
        self.gedeeltelijk = gedeeltelijk


def stream_antwoord(client: Client, p: Partner, inst: dict[str, Any],
                    extra: list[str] | None = None) -> str:
    """
    Vraag een antwoord aan het model van partner p, op basis van zijn
    volledige geschiedenis, en toon het token per token in de console.

    'extra' zijn regie-instructies (rondeteller, herkansing) die enkel voor
    deze ene oproep aan het laatste bericht worden toegevoegd. De bewaarde
    geschiedenis blijft daardoor zuiver.
    """
    berichten = [dict(m) for m in p.geschiedenis]
    if extra:
        berichten[-1]["content"] = berichten[-1]["content"] + "\n\n" + "\n".join(extra)

    kwargs: dict[str, Any] = {
        "model": p.model,
        "messages": berichten,
        "stream": True,
        "options": {
            "temperature": inst["temperatuur"],
            "num_ctx": inst["num_ctx"],
            "repeat_penalty": inst["repeat_penalty"],
            "repeat_last_n": inst["repeat_last_n"],
        },
    }
    if p.kan_denken:
        kwargs["think"] = False  # geen interne 'denk'-tekst, enkel het antwoord

    stukken: list[str] = []
    try:
        for chunk in client.chat(**kwargs):
            deel = chunk.message.content or ""
            if deel:
                stukken.append(deel)
                print(deel, end="", flush=True)
    except KeyboardInterrupt:
        print(flush=True)
        raise OnderbrokenTijdensBeurt(THINK_TAG.sub("", "".join(stukken)).strip())
    print("\n", flush=True)
    return THINK_TAG.sub("", "".join(stukken)).strip()


def toon_kop(p: Partner, ronde: int, totaal: int) -> None:
    print(f"{p.kleur}{VET}[{ronde}/{totaal}] {p.label} — {p.naam}{RESET} "
          f"{KLEUR_INFO}({p.model}, {nu():%H:%M:%S}){RESET}")
    print(p.kleur, end="", flush=True)


# ---------------------------------------------------------------------------
# Debat uitvoeren
# ---------------------------------------------------------------------------

@dataclass
class Resultaat:
    status: str
    berichten: int
    rondes: int
    log: Path
    herhalingen: int = 0
    herkansingen: int = 0
    weigeringen: int = 0
    overgeslagen: int = 0


def voer_debat_uit(client: Client, debat: Debat, inst: dict[str, Any],
                   modellen: list[str]) -> Resultaat:
    """Voer één volledig debat uit. Geeft het resultaat terug."""
    # 1. Modellen controleren en partners opbouwen.
    model_a = model_oplossen(inst["model_a"], modellen)
    model_b = model_oplossen(inst["model_b"], modellen)
    a = Partner("A", debat.partner_a["naam"], debat.partner_a["identiteit"],
                debat.partner_a["standpunt"], model_a, KLEUR_A, kan_denken(client, model_a))
    b = Partner("B", debat.partner_b["naam"], debat.partner_b["identiteit"],
                debat.partner_b["standpunt"], model_b, KLEUR_B, kan_denken(client, model_b))
    if not (a.standpunt and b.standpunt):
        info("ℹ️  Tip: geef beide partners een 'standpunt' in info.json, "
             "anders gaan de modellen snel akkoord met elkaar.")

    # 2. Elke partner krijgt een eigen systeemprompt; A opent het debat.
    a.geschiedenis.append({"role": "system", "content": systeemprompt(a, b, debat, inst)})
    a.geschiedenis.append({"role": "user", "content": openingsbericht(debat)})
    b.geschiedenis.append({"role": "system", "content": systeemprompt(b, a, debat, inst)})

    # 3. Log aanmaken.
    start = nu()
    log = MarkdownLog(SCRIPT_DIR / inst["logmap"], debat, start)
    log.kop(debat, a, b, start)
    transcript: list[tuple[int, Partner, str]] = []   # (ronde, spreker, tekst)

    print(f"\n{VET}{'═' * 70}\n🎙️  {debat.titel}\n{'═' * 70}{RESET}")
    info(f"A: {a.naam} ({a.model})  ·  B: {b.naam} ({b.model})  ·  {debat.rondes} rondes")
    info(f"Log: {log.pad}\n")

    berichten = 0
    voltooide_rondes = 0
    herhalingen = 0
    herkansingen = 0
    weigeringen = 0
    overgeslagen = 0
    status = "✅ Voltooid"

    try:
        for ronde in range(1, debat.rondes + 1):
            log.ronde(ronde)
            # Om beurt: eerst A, dan B.
            for spreker, luisteraar in ((a, b), (b, a)):
                toon_kop(spreker, ronde, debat.rondes)
                vermijd = (openingen_en_afsluiters(spreker, inst["openingen_vermijden"])
                           if inst["openingen_vermijden"] > 0 else [])
                regie = ([beurtinstructie(ronde, debat.rondes, spreker, vermijd)]
                         if inst["rondeteller"] else [])
                opmerkingen: list[str] = []
                overslaan = False

                def opnieuw(reden: str, extra: str) -> str:
                    """Toon een melding en vraag één nieuwe poging met extra regie."""
                    print(RESET, end="")
                    info(f"{reden} Nieuwe poging…")
                    print(spreker.kleur, end="", flush=True)
                    return stream_antwoord(client, spreker, inst, regie + [extra])

                try:
                    tekst = stream_antwoord(client, spreker, inst, regie)

                    # 1. Weigering / rolbreuk ("Ik kan dit debat niet voortzetten…").
                    if is_weigering(tekst):
                        weigeringen += 1
                        if inst["weigering_herkansing"]:
                            herkansingen += 1
                            tekst = opnieuw("🚫 Het model weigert of valt uit zijn rol.",
                                            WEIGERING_HERKANSING)
                        if is_weigering(tekst):
                            # Niet in de geschiedenis: anders blijft het model
                            # zijn eigen weigering herhalen.
                            overslaan = True
                            overgeslagen += 1
                            opmerkingen.append("weigering, overgeslagen")
                        else:
                            opmerkingen.append("herkansing na weigering")

                    # 2. Herhaling: vergelijk met de laatste N berichten.
                    if not overslaan:
                        eerdere = [t for _, _, t in transcript[-inst["herhaling_vergelijk"]:]]
                        score = gelijkenis(tekst, eerdere)
                        if score >= inst["herhaling_drempel"]:
                            herhalingen += 1
                            if inst["herhaling_herkansing"]:
                                herkansingen += 1
                                tekst2 = opnieuw(f"♻️  Lijkt {score:.0%} op een eerder bericht.",
                                                 HERKANSING)
                                if is_weigering(tekst2):
                                    weigeringen += 1
                                    opmerkingen.append(f"herhaling ({score:.0%}), herkansing was weigering")
                                else:
                                    tekst = tekst2
                                    score2 = gelijkenis(tekst, eerdere)
                                    opmerkingen.append(
                                        f"herkansing na herhaling ({score:.0%})"
                                        if score2 < inst["herhaling_drempel"]
                                        else f"herhaling, ook na herkansing ({score2:.0%})")
                            else:
                                opmerkingen.append(f"herhaling ({score:.0%})")
                except OnderbrokenTijdensBeurt as e:
                    if e.gedeeltelijk:
                        log.bericht(spreker, e.gedeeltelijk, nu(), onderbroken=True)
                        transcript.append((ronde, spreker, e.gedeeltelijk))
                        berichten += 1
                    raise KeyboardInterrupt from None
                print(RESET, end="")

                if not tekst:
                    tekst = "_(geen antwoord ontvangen)_"

                if overslaan:
                    # Beurt overslaan: wel loggen, niet in de geschiedenis of samenvatting.
                    info(f"⏭️  Beurt van {spreker.naam} overgeslagen.")
                    voeg_toe(luisteraar.geschiedenis, "user",
                             f"[Regie: {spreker.naam} slaat deze beurt over. "
                             f"Ga verder met je volgende argument.]")
                    log.bericht(spreker, tekst, nu(), opmerking="; ".join(opmerkingen))
                    continue

                # Volledige context bewaren:
                # - voor de spreker is het zijn eigen antwoord (assistant);
                # - voor de luisteraar is het een binnenkomend bericht (user).
                voeg_toe(spreker.geschiedenis, "assistant", tekst)
                voeg_toe(luisteraar.geschiedenis, "user", tekst)

                log.bericht(spreker, tekst, nu(), opmerking="; ".join(opmerkingen))
                transcript.append((ronde, spreker, tekst))
                berichten += 1
            voltooide_rondes = ronde

    except KeyboardInterrupt:
        print(RESET)
        info("\n⏸️  Onderbroken met Ctrl+C. Het gesprek tot nu toe is bewaard.")
        status = "⏸️ Onderbroken door gebruiker"
    except ResponseError as e:
        print(RESET)
        fout(f"Ollama gaf een fout: {e.error} (status {e.status_code})")
        status = f"❌ Fout: {e.error}"
    except ConnectionError:
        print(RESET)
        fout("Verbinding met Ollama verbroken tijdens het debat.")
        status = "❌ Fout: verbinding met Ollama verbroken"

    # 4. Samenvatting (enkel bij een voltooid debat, of als er iets te vatten is
    #    na een fout; bij Ctrl+C slaan we ze over om de gebruiker niet te laten wachten).
    if inst["samenvatting"] and transcript and not status.startswith("⏸️"):
        maak_samenvatting(client, debat, transcript, inst, modellen, log)

    einde = nu()
    log.afsluiting(status, berichten, voltooide_rondes, start, einde, herhalingen, herkansingen,
                   weigeringen, overgeslagen)
    return Resultaat(status, berichten, voltooide_rondes, log.pad, herhalingen, herkansingen,
                     weigeringen, overgeslagen)


def _vraag_model(client: Client, model: str, prompt: str, inst: dict[str, Any],
                 tonen: bool) -> str:
    """
    Eén losse vraag aan een model (zonder gespreksgeschiedenis).
    Met tonen=True wordt het antwoord live gestreamd naar de console.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "options": {"temperature": 0.3, "num_ctx": inst["num_ctx"]},
    }
    if kan_denken(client, model):
        kwargs["think"] = False

    stukken: list[str] = []
    for chunk in client.chat(**kwargs):
        deel = chunk.message.content or ""
        stukken.append(deel)
        if tonen:
            print(deel, end="", flush=True)
    if tonen:
        print("\n")
    return THINK_TAG.sub("", "".join(stukken)).strip()


def maak_samenvatting(client: Client, debat: Debat, transcript: list[tuple[int, Partner, str]],
                      inst: dict[str, Any], modellen: list[str], log: MarkdownLog) -> None:
    """
    Vat het debat samen in twee stappen (map-reduce), zodat het nooit
    groter wordt dan het contextvenster (num_ctx):

    1. Deelsamenvatting per blok van N rondes (instelling 'samenvatting_blok').
    2. Eindsamenvatting op basis van die deelsamenvattingen.

    Is het debat korter dan één blok, dan wordt het in één keer samengevat.
    """
    try:
        model = model_oplossen(inst["samenvatting_model"] or inst["model_a"], modellen)
    except ModelFout as e:
        fout(f"Samenvatting overgeslagen: {e}")
        return

    # Transcript opdelen in blokken van N rondes.
    n = inst["samenvatting_blok"]
    blokken: dict[int, list[tuple[int, Partner, str]]] = {}
    for item in transcript:
        blokken.setdefault((item[0] - 1) // n, []).append(item)

    def als_tekst(items: list[tuple[int, Partner, str]]) -> str:
        return "\n\n".join(f"[Ronde {r}] {p.naam}: {t}" for r, p, t in items)

    eindopdracht = (
        f"Maak in het {inst['taal']} een beknopte samenvatting in Markdown met:\n"
        f"1. De kernstandpunten van elke deelnemer (kopjes per deelnemer).\n"
        f"2. De sterkste argumenten en tegenargumenten.\n"
        f"3. Hoe het debat evolueerde: verschoven standpunten of toegevingen.\n"
        f"4. Punten van overeenstemming en blijvende meningsverschillen.\n"
        f"5. Een neutrale conclusie in 2–3 zinnen.\n"
        f"Gebruik geen titel van niveau 1 of 2."
    )

    delen: list[tuple[str, str]] = []
    try:
        if len(blokken) > 1:
            # Stap 1: deelsamenvattingen (niet gestreamd, enkel voortgang tonen).
            print(f"\n{VET}📝 Samenvatting ({model}) — {len(blokken)} blokken van {n} rondes{RESET}")
            for i, (_, items) in enumerate(sorted(blokken.items()), start=1):
                van, tot = items[0][0], items[-1][0]
                titel = f"Rondes {van}–{tot}"
                info(f"   Blok {i}/{len(blokken)}: {titel} …")
                prompt = (
                    f"Hieronder staat een deel ({titel}) van een debat over: {debat.onderwerp}\n\n"
                    f"Vat dit deel in het {inst['taal']} samen in maximaal 150 woorden: "
                    f"welke argumenten bracht elke deelnemer, en waar reageerden ze op elkaar? "
                    f"Gebruik geen titels.\n\n--- DEBAT ---\n{als_tekst(items)}"
                )
                delen.append((titel, _vraag_model(client, model, prompt, inst, tonen=False)))

            # Stap 2: eindsamenvatting op basis van de deelsamenvattingen.
            info("   Eindsamenvatting …\n")
            bron = "\n\n".join(f"### {t}\n{inhoud}" for t, inhoud in delen)
            prompt = (
                f"Hieronder staan deelsamenvattingen, in chronologische volgorde, van een "
                f"debat over: {debat.onderwerp}\n\n{eindopdracht}\n\n"
                f"--- DEELSAMENVATTINGEN ---\n{bron}"
            )
        else:
            # Kort debat: in één keer samenvatten.
            print(f"\n{VET}📝 Samenvatting ({model}){RESET}")
            prompt = (
                f"Hieronder staat een debat over: {debat.onderwerp}\n\n{eindopdracht}"
                f"\n\n--- DEBAT ---\n{als_tekst(transcript)}"
            )
        tekst = _vraag_model(client, model, prompt, inst, tonen=True)

    except KeyboardInterrupt:
        print()
        info("Samenvatting onderbroken; reeds gemaakte deelsamenvattingen worden bewaard.")
        tekst = "_(Eindsamenvatting onderbroken)_"
    except (ResponseError, ConnectionError) as e:
        fout(f"Samenvatting mislukt: {e}")
        tekst = f"_(Eindsamenvatting mislukt: {e})_"

    if delen or not tekst.startswith("_("):
        log.samenvatting(model, tekst, delen)


# ---------------------------------------------------------------------------
# Hoofdprogramma
# ---------------------------------------------------------------------------

def argumenten() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TalkingAI — laat twee lokale Ollama-modellen met elkaar debatteren.")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG,
                        help="pad naar de configuratie (standaard: info.json naast het script)")
    parser.add_argument("-d", "--debat", action="append", metavar="ID",
                        help="voer enkel dit debat uit (mag meermaals)")
    parser.add_argument("-r", "--rondes", type=int,
                        help="overschrijf het aantal rondes (handig om te testen)")
    parser.add_argument("-l", "--lijst", action="store_true",
                        help="toon de debatten uit de configuratie en stop")
    return parser.parse_args()


def main() -> int:
    console_voorbereiden()
    args = argumenten()

    # Configuratie laden.
    try:
        inst, debatten = config_laden(args.config)
    except ConfigFout as e:
        fout(str(e))
        return 2

    if args.lijst:
        print(f"{VET}Debatten in {args.config.name}:{RESET}")
        for d in debatten:
            print(f"  - {d.id:<20} {d.titel}  ({d.rondes} rondes)")
        return 0

    # Selectie van debatten.
    if args.debat:
        onbekend = [i for i in args.debat if i not in {d.id for d in debatten}]
        if onbekend:
            fout(f"Onbekend debat-id: {', '.join(onbekend)}. Gebruik --lijst om ze te zien.")
            return 2
        debatten = [d for d in debatten if d.id in args.debat]
    if args.rondes is not None:
        if args.rondes < 1:
            fout("--rondes moet minstens 1 zijn.")
            return 2
        for d in debatten:
            d.rondes = args.rondes

    # Ollama en modellen controleren vóór we beginnen.
    client = Client(host=inst["ollama_host"])
    try:
        modellen = geinstalleerde_modellen(client)
        model_oplossen(inst["model_a"], modellen)
        model_oplossen(inst["model_b"], modellen)
    except ModelFout as e:
        fout(str(e))
        return 3

    # Debatten één voor één uitvoeren.
    resultaten: list[tuple[Debat, Resultaat]] = []
    for debat in debatten:
        try:
            res = voer_debat_uit(client, debat, inst, modellen)
        except ModelFout as e:
            fout(str(e))
            return 3
        resultaten.append((debat, res))
        print(f"\n{KLEUR_OK}{VET}{res.status} — {res.berichten} berichten uitgewisseld "
              f"in {res.rondes} rondes.{RESET}")
        if res.herhalingen or res.weigeringen:
            info(f"♻️  {res.herhalingen} herhaling(en), 🚫 {res.weigeringen} weigering(en), "
                 f"{res.herkansingen} herkansing(en), {res.overgeslagen} beurt(en) overgeslagen.")
        info(f"Opgeslagen in: {res.log}")
        if res.status.startswith("⏸️"):
            break  # Ctrl+C stopt ook de resterende debatten

    # Eindoverzicht.
    totaal = sum(r.berichten for _, r in resultaten)
    print(f"\n{VET}{'─' * 70}\nKlaar: {len(resultaten)} debat(ten), "
          f"{totaal} berichten in totaal.{RESET}")
    return 0 if all(r.status.startswith("✅") for _, r in resultaten) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:  # Ctrl+C buiten een debat (bv. tijdens opstarten)
        print(RESET)
        info("Afgebroken.")
        sys.exit(130)
