# TalkingAI 🗨️

**Twee AI-modellen die met elkaar in gesprek gaan, volledig lokaal op mijn eigen pc. Geen cloud en geen abonnement.**

TalkingAI is een klein Python-project waarin twee lokale taalmodellen (LLM's) met elkaar converseren. Ik start het script en geef een onderwerp mee, daarna nemen de modellen het over. Wat gebeurt er als twee AI's zonder menselijke tussenkomst met elkaar praten? En hoe ver kom je daarmee op gewone consumentenhardware?

> 🚧 **Status:** work in progress. Het script werkt en de eerste testdebatten zijn gevoerd.

> [!WARNING]
> **Entertainment, geen advies.** De debatten zijn volledig gegenereerd door kleine lokale taalmodellen. Die klinken overtuigend, maar verzinnen vlot cijfers, "cases" en zelfs wetsartikelen. In één testdebat over AI en ondernemen haalden ze er hypotheekleningen bij en verwezen ze naar onbestaande wetsartikelen. De inhoud is dus moeilijk bruikbaar als informatie. Gebruik niets hiervan als financieel, juridisch of ander advies: **lezen en gebruiken op eigen risico.**

---

## Over dit project

TalkingAI is in de eerste plaats een **leerproject**: ik wil ontdekken wat taalmodellen vandaag kunnen, en dat langs twee kanten.

- **Lokaal:** twee open modellen (`llama3.1` en `qwen3`) draaien op mijn eigen pc via Ollama. Hoe ver kom je met een gewone gaming-GPU? Waar liggen de grenzen van kleine modellen?
- **Als assistent:** het script, de documentatie en de analyse van de debatlogs maakte ik samen met **Claude** (Anthropic), in Cowork. Ik bracht het idee, de vereisten, de tests en de keuzes. Claude schreef de code en hielp verklaren waarom de debatten soms ontspoorden.

Net dat contrast maakt het interessant: een groot model helpt een kleine opstelling te bouwen, en de kleine modellen tonen meteen waar de grenzen van taalmodellen liggen.

---

## Waarom lokaal?

- **Privacy:** het hele gesprek blijft op mijn eigen machine.
- **Kosten:** twee modellen die eindeloos praten zouden via een API snel geld kosten. Lokaal is het gratis.
- **Leerwaarde:** je ziet zelf wat een model nodig heeft (VRAM, modelgrootte, kwantisatie) en welke afwegingen je moet maken.

## Hoe het werkt

```
Ik: start + onderwerp
        ↓
   [Python-script]  ←→  llama3.1:8b  (A)
          ↕
       qwen3:8b     (B)
```

1. Ik start het script. Onderwerp en identiteiten van beide partners staan in `info.json`.
2. Het script geeft het startbericht aan model A.
3. Het antwoord van A gaat naar B, het antwoord van B terug naar A, enzovoort.
4. Na het ingestelde aantal rondes stopt het gesprek en volgt een samenvatting.

Een debat van 50 rondes past niet volledig in het geheugen van een 8B-model. Daarom wordt het eerst per blok van 10 rondes samengevat, en daarna volgt een eindsamenvatting.

Beide modellen draaien via [Ollama](https://ollama.com) op `localhost`. Mijn rol is *human-in-the-loop*: ik zet het gesprek op gang en kijk daarna toe.

## Tech stack

| Onderdeel | Keuze |
|---|---|
| Model-runtime | Ollama 0.40.5 |
| Modellen | `llama3.1:8b` (gesprekspartner A), `qwen3:8b` (gesprekspartner B) |
| Taal | Python 3 + [`ollama`](https://github.com/ollama/ollama-python) 0.6.3 |
| Hardware | NVIDIA RTX 3070 (8 GB VRAM) |
| Ontwikkelassistent | Claude (Anthropic), via Cowork: code, documentatie, loganalyse |

### Waarom 8B-modellen?

Mijn GPU heeft 8 GB VRAM. Een 8B-model in 4-bit kwantisatie (± 5 GB) past daar volledig in en blijft dus snel. Twee modellen samen passen er niet tegelijk in, dus Ollama wisselt ze om beurten. Dat is trager, maar wel haalbaar. Met grotere modellen zou elk gesprek traag als stroop worden.

## Aan de slag

**Vereisten:** [Ollama](https://ollama.com/download), Python 3.10+ en een GPU met ±8 GB VRAM.

```bash
ollama pull llama3.1:8b
ollama pull qwen3:8b
pip install -r requirements.txt
```

### Een discussie starten

1. **Kies of maak een onderwerp** in `info.json`. Elk debat heeft een `id`, een `onderwerp` en twee partners. Een partner kan één woord zijn of een volledige beschrijving:

   ```json
   {
     "id": "koffie-thee",
     "onderwerp": "Stelling: koffie is beter dan thee.",
     "partner_a": { "naam": "Barista", "identiteit": "Een gepassioneerde barista.", "standpunt": "Voor de stelling" },
     "partner_b": { "naam": "Theemeester", "identiteit": "Een rustige theekenner.", "standpunt": "Tegen de stelling" }
   }
   ```

   Tip: geef elke partner een tegengesteld `standpunt`. Anders worden kleine modellen het na een paar rondes gewoon met elkaar eens.

2. **Start het debat:**

   ```bash
   python talkingai.py --lijst                       # welke debatten zijn er?
   python talkingai.py --debat koffie-thee --rondes 3  # korte test
   python talkingai.py --debat koffie-thee             # volledig debat (50 rondes)
   ```

3. **Kijk mee.** De berichten verschijnen live in de console. Het volledige debat, met een automatische samenvatting, wordt opgeslagen in `logs/` als Markdown-bestand met datum en tijd in de naam. Stoppen kan altijd met Ctrl+C: wat al gezegd is, blijft bewaard.

Instellingen zoals het aantal rondes, de taal of de lengte van een beurt staan ook in `info.json`, onder `instellingen`.

## Wat ik onderweg leerde

Kleine modellen debatteren graag, maar zijn véél te beleefd. Zonder vast standpunt waren ze na vijf rondes "het volledig met elkaar eens", na zeven rondes namen ze afscheid, en daarna begonnen ze elkaar letterlijk te herhalen. De oplossing kwam uit vier hoeken: elke partner een vast standpunt, een "regisseur" die bij elke beurt zegt in welke ronde we zitten, herhalingsdetectie met een herkansing, en een iets strengere herhalingsstraf in Ollama.

Daarna volgde een tweede les: een model dat één keer weigert ("Ik kan dit debat niet voortzetten"), blijft dat herhalen zolang die weigering in zijn eigen geschiedenis staat. Het script herkent zulke weigeringen nu en houdt ze uit het geheugen van het model. Het merkt ook vaste zinspatronen op ("Je benadrukt… maar vergeet dat…") en vraagt het model om anders te beginnen. En de belangrijkste les: kleine modellen verzinnen vlot cijfers, voorbeelden en zelfs wetsartikelen, en brengen ze met grote stelligheid. Een AI-debat is dus entertainment en denkvoer, geen bron.

## Roadmap

- [x] Ollama opzetten
- [x] Lokale modellen kiezen en downloaden
- [x] Python-omgeving klaarzetten
- [x] Python-script dat twee modellen met elkaar laat praten
- [x] Eerste testdebat (3 rondes)
- [ ] Volledig debat van 50 rondes
- [x] Bevindingen delen

## Licentie

De code van dit project valt onder de [MIT-licentie](LICENSE): je mag ze vrij gebruiken, kopiëren, aanpassen en verspreiden, zolang de licentietekst erbij blijft. De software wordt geleverd "as is", zonder enige garantie.

De taalmodellen zelf zitten niet in deze repo en hebben hun eigen licentie: [Llama 3.1](https://www.llama.com/llama3_1/license/) valt onder de Llama 3.1 Community License van Meta, [Qwen3](https://ollama.com/library/qwen3) onder Apache 2.0.

---

*Gemaakt uit nieuwsgierigheid en als leerproject.* 🚀
