# Copyright 2025-2026 Valentin Noël
# SPDX-License-Identifier: Apache-2.0

"""The sentence set of the passive-voice use case, written to
``examples/use_cases/data/passive_active.json``.

Groups (``group`` field):

- ``en_pair``: 72 matched English active/passive pairs. 36 (agent, verb,
  patient) triples, each in a short template, and half of them with
  adjectives (T2) or with a time adverb and a place (T3). Active and passive
  start with the same word ("The" or "Yesterday"), so the first token, which
  is the attention sink of models without a BOS token (Phi-3), is the same.
- ``en_firstword``: 24 pairs in the form of the original example ("The A
  verbed a new P." / "A new P was verbed by the A."): the passive starts
  with "A", the active with "The".
- ``en_by_active``: 24 actives with a non-agentive "by" phrase ("The cat
  slept by the door."), paired with no passive.
- ``en_agentless``: 24 agentless passives ("The cat was chased."), each
  paired with the short active of the same triple.
- ``fr_pair``, ``de_pair``: 24 French and 24 German active/passive pairs.
- ``original``: a reference pair in the form of ``en_firstword`` ("The"
  against "A" as first word).

Each item: ``id``, ``group``, ``pair`` (pair id, shared by the active and
passive of a pair), ``voice`` (``active``, ``passive``, ``by_active``,
``agentless``), ``lang``, ``text``, and for English ``agent``/``patient``.

    python examples/use_cases/passive_data.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "passive_active.json"

# agent, past, participle, patient, agent adjective, patient adjective, place
TRIPLES = [
    ("scientist", "discovered", "discovered", "particle", "young", "new", "laboratory"),
    ("chef", "cooked", "cooked", "meal", "tired", "delicious", "kitchen"),
    ("dog", "chased", "chased", "cat", "big", "small", "garden"),
    ("boy", "threw", "thrown", "ball", "little", "red", "park"),
    ("woman", "drove", "driven", "car", "old", "blue", "city"),
    ("man", "ate", "eaten", "apple", "hungry", "green", "market"),
    ("bird", "built", "built", "nest", "small", "tiny", "forest"),
    ("teacher", "graded", "graded", "exam", "strict", "final", "school"),
    ("artist", "painted", "painted", "portrait", "famous", "large", "studio"),
    ("author", "wrote", "written", "novel", "young", "long", "cottage"),
    ("engineer", "designed", "designed", "bridge", "clever", "steel", "river"),
    ("officer", "arrested", "arrested", "thief", "brave", "nervous", "station"),
    ("doctor", "treated", "treated", "patient", "kind", "elderly", "hospital"),
    ("storm", "destroyed", "destroyed", "village", "violent", "quiet", "coast"),
    ("company", "hired", "hired", "manager", "large", "new", "office"),
    ("committee", "approved", "approved", "plan", "local", "ambitious", "town"),
    ("farmer", "planted", "planted", "tree", "old", "young", "field"),
    ("pilot", "landed", "landed", "plane", "experienced", "damaged", "airport"),
    ("child", "broke", "broken", "window", "curious", "glass", "house"),
    ("judge", "sentenced", "sentenced", "criminal", "stern", "dangerous", "court"),
    ("student", "solved", "solved", "problem", "bright", "difficult", "library"),
    ("composer", "composed", "composed", "symphony", "gifted", "beautiful", "theatre"),
    ("mechanic", "repaired", "repaired", "engine", "skilled", "broken", "garage"),
    ("cat", "caught", "caught", "mouse", "black", "grey", "barn"),
    ("gardener", "watered", "watered", "garden", "patient", "dry", "estate"),
    ("director", "filmed", "filmed", "scene", "famous", "final", "desert"),
    ("baker", "baked", "baked", "cake", "busy", "chocolate", "bakery"),
    ("editor", "reviewed", "reviewed", "article", "careful", "long", "newsroom"),
    ("army", "defeated", "defeated", "enemy", "small", "large", "valley"),
    ("nurse", "helped", "helped", "visitor", "kind", "lost", "clinic"),
    ("architect", "drew", "drawn", "blueprint", "talented", "detailed", "harbour"),
    ("queen", "opened", "opened", "museum", "young", "new", "capital"),
    ("hunter", "shot", "shot", "deer", "old", "young", "mountains"),
    ("senator", "proposed", "proposed", "law", "popular", "new", "parliament"),
    ("programmer", "fixed", "fixed", "bug", "junior", "subtle", "campus"),
    ("photographer", "took", "taken", "picture", "famous", "stunning", "beach"),
]

BY_ACTIVES = [
    "The cat slept by the door.", "The man waited by the window.",
    "The children played by the river.", "The dog sat by the fire.",
    "The woman stood by the gate.", "The boat drifted by the shore.",
    "The girl read by the lamp.", "The soldier stood by the wall.",
    "The old man rested by the tree.", "The students gathered by the fountain.",
    "The car stopped by the station.", "The horse drank by the stream.",
    "The fisherman sat by the lake.", "The couple danced by the sea.",
    "The guard waited by the entrance.", "The bird landed by the pond.",
    "The family camped by the forest.", "The boy sat by the window.",
    "The train passed by the farm.", "The runner rested by the bench.",
    "The artist painted by the harbour.", "The teacher stood by the board.",
    "The visitor waited by the counter.", "The deer grazed by the road.",
]

FR_PAIRS = [
    ("Le scientifique a découvert la particule.", "La particule a été découverte par le scientifique."),
    ("Le chef a préparé le repas.", "Le repas a été préparé par le chef."),
    ("Le chien a poursuivi le chat.", "Le chat a été poursuivi par le chien."),
    ("Le garçon a lancé le ballon.", "Le ballon a été lancé par le garçon."),
    ("La femme a conduit la voiture.", "La voiture a été conduite par la femme."),
    ("L'homme a mangé la pomme.", "La pomme a été mangée par l'homme."),
    ("L'oiseau a construit le nid.", "Le nid a été construit par l'oiseau."),
    ("Le professeur a corrigé l'examen.", "L'examen a été corrigé par le professeur."),
    ("L'artiste a peint le portrait.", "Le portrait a été peint par l'artiste."),
    ("L'auteur a écrit le roman.", "Le roman a été écrit par l'auteur."),
    ("L'ingénieur a conçu le pont.", "Le pont a été conçu par l'ingénieur."),
    ("Le médecin a soigné le patient.", "Le patient a été soigné par le médecin."),
    ("La tempête a détruit le village.", "Le village a été détruit par la tempête."),
    ("L'entreprise a embauché le directeur.", "Le directeur a été embauché par l'entreprise."),
    ("Le comité a approuvé le projet.", "Le projet a été approuvé par le comité."),
    ("Le fermier a planté l'arbre.", "L'arbre a été planté par le fermier."),
    ("L'enfant a cassé la fenêtre.", "La fenêtre a été cassée par l'enfant."),
    ("Le juge a condamné le criminel.", "Le criminel a été condamné par le juge."),
    ("L'étudiant a résolu le problème.", "Le problème a été résolu par l'étudiant."),
    ("Le mécanicien a réparé le moteur.", "Le moteur a été réparé par le mécanicien."),
    ("Le chat a attrapé la souris.", "La souris a été attrapée par le chat."),
    ("Le boulanger a cuit le gâteau.", "Le gâteau a été cuit par le boulanger."),
    ("L'éditeur a relu l'article.", "L'article a été relu par l'éditeur."),
    ("Le programmeur a corrigé le bogue.", "Le bogue a été corrigé par le programmeur."),
]

DE_PAIRS = [
    ("Der Wissenschaftler entdeckte das Teilchen.", "Das Teilchen wurde vom Wissenschaftler entdeckt."),
    ("Der Koch kochte das Essen.", "Das Essen wurde vom Koch gekocht."),
    ("Der Hund jagte die Katze.", "Die Katze wurde vom Hund gejagt."),
    ("Der Junge warf den Ball.", "Der Ball wurde vom Jungen geworfen."),
    ("Die Frau fuhr das Auto.", "Das Auto wurde von der Frau gefahren."),
    ("Der Mann aß den Apfel.", "Der Apfel wurde vom Mann gegessen."),
    ("Der Vogel baute das Nest.", "Das Nest wurde vom Vogel gebaut."),
    ("Der Lehrer korrigierte die Prüfung.", "Die Prüfung wurde vom Lehrer korrigiert."),
    ("Der Künstler malte das Porträt.", "Das Porträt wurde vom Künstler gemalt."),
    ("Der Autor schrieb den Roman.", "Der Roman wurde vom Autor geschrieben."),
    ("Der Ingenieur entwarf die Brücke.", "Die Brücke wurde vom Ingenieur entworfen."),
    ("Der Arzt behandelte den Patienten.", "Der Patient wurde vom Arzt behandelt."),
    ("Der Sturm zerstörte das Dorf.", "Das Dorf wurde vom Sturm zerstört."),
    ("Die Firma kaufte das Gebäude.", "Das Gebäude wurde von der Firma gekauft."),
    ("Der Ausschuss genehmigte den Plan.", "Der Plan wurde vom Ausschuss genehmigt."),
    ("Der Bauer pflanzte den Baum.", "Der Baum wurde vom Bauern gepflanzt."),
    ("Das Kind zerbrach das Fenster.", "Das Fenster wurde vom Kind zerbrochen."),
    ("Der Richter verurteilte den Verbrecher.", "Der Verbrecher wurde vom Richter verurteilt."),
    ("Der Student löste das Problem.", "Das Problem wurde vom Studenten gelöst."),
    ("Der Mechaniker reparierte den Motor.", "Der Motor wurde vom Mechaniker repariert."),
    ("Die Katze fing die Maus.", "Die Maus wurde von der Katze gefangen."),
    ("Der Bäcker backte den Kuchen.", "Der Kuchen wurde vom Bäcker gebacken."),
    ("Der Redakteur prüfte den Artikel.", "Der Artikel wurde vom Redakteur geprüft."),
    ("Der Programmierer behob den Fehler.", "Der Fehler wurde vom Programmierer behoben."),
]

ORIGINAL = ("The scientist discovered a new particle.",
            "A new particle was discovered by the scientist.")


def build() -> list:
    items = []

    def add(group, pair, voice, lang, text, **extra):
        items.append(dict(id=f"{pair}_{voice}", group=group, pair=pair, voice=voice,
                          lang=lang, text=text, **extra))

    for k, (a, past, pp, p, adj_a, adj_p, place) in enumerate(TRIPLES):
        act, pas = dict(agent=a, patient=p, verb=past), dict(agent=a, patient=p, verb=pp)
        add("en_pair", f"en_t1_{k:02d}", "active", "en", f"The {a} {past} the {p}.", **act)
        add("en_pair", f"en_t1_{k:02d}", "passive", "en", f"The {p} was {pp} by the {a}.", **pas)
        if k < 18:
            add("en_pair", f"en_t2_{k:02d}", "active", "en",
                f"The {adj_a} {a} {past} the {adj_p} {p}.", **act)
            add("en_pair", f"en_t2_{k:02d}", "passive", "en",
                f"The {adj_p} {p} was {pp} by the {adj_a} {a}.", **pas)
        else:
            add("en_pair", f"en_t3_{k:02d}", "active", "en",
                f"Yesterday the {a} {past} the {p} near the {place}.", **act)
            add("en_pair", f"en_t3_{k:02d}", "passive", "en",
                f"Yesterday the {p} was {pp} by the {a} near the {place}.", **pas)
    for k, (a, past, pp, p, adj_a, adj_p, place) in enumerate(TRIPLES[:24]):
        act, pas = dict(agent=a, patient=p, verb=past), dict(agent=a, patient=p, verb=pp)
        add("en_firstword", f"en_fw_{k:02d}", "active", "en",
            f"The {a} {past} a {adj_p} {p}.", **act)
        add("en_firstword", f"en_fw_{k:02d}", "passive", "en",
            f"A {adj_p} {p} was {pp} by the {a}.", **pas)
    for k, (a, past, pp, p, *_rest) in enumerate(TRIPLES[:24]):
        act, pas = dict(agent=a, patient=p, verb=past), dict(agent=a, patient=p, verb=pp)
        add("en_agentless", f"en_al_{k:02d}", "active", "en", f"The {a} {past} the {p}.", **act)
        add("en_agentless", f"en_al_{k:02d}", "agentless", "en", f"The {p} was {pp}.", **pas)
    for k, text in enumerate(BY_ACTIVES):
        add("en_by_active", f"en_by_{k:02d}", "by_active", "en", text)
    for lang, pairs in (("fr", FR_PAIRS), ("de", DE_PAIRS)):
        for k, (act, pas) in enumerate(pairs):
            add(f"{lang}_pair", f"{lang}_{k:02d}", "active", lang, act)
            add(f"{lang}_pair", f"{lang}_{k:02d}", "passive", lang, pas)
    add("original", "original", "active", "en", ORIGINAL[0], agent="scientist",
        patient="particle", verb="discovered")
    add("original", "original", "passive", "en", ORIGINAL[1], agent="scientist",
        patient="particle", verb="discovered")
    return items


def load() -> list:
    with open(OUT, encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    items = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=1, ensure_ascii=False)
    from collections import Counter
    print(OUT, len(items), dict(Counter((i["group"], i["voice"]) for i in items)))
