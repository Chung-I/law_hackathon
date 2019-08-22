from typing import Dict, Tuple, List, Any, Union
import json
import requests
import numpy as np
from graphviz import Digraph
from collections import Counter
import benepar
from pywordseg import Wordseg
import spacy
from opencc import OpenCC
from nltk import ParentedTree
import pdb
import stanfordnlp
from stanfordnlp.server import CoreNLPClient
import urllib
from functools import partial

DEIXIS = ["他", "她", "其"]

def get_span(obj):
    return (obj['characterOffsetBegin']-5, obj['characterOffsetEnd']-5)

OBJECTS = {}
SUBJECTS = {}

def line_break(string, interval):
    return '\\n'.join([string[idx:(idx+interval)] for idx in range(0,len(string),interval)])

def is_included_in(a, b, portion=0.0):
    # a and b are spans. portion < len(a)/len(b)
    return b[0] <= a[0] and a[-1] <= b[-1] and float(len(b)) * portion <= float(len(a))

def mention_has_intersection(a, b, portion=0.0):
    sub_func = lambda a, b: (a in b and float(len(b)) * portion <= float(len(a)))
    result = sub_func(a, b) or sub_func(b, a)
    return result
    #print(f"a: {a}, b: {b}, result: {result}")

def print_mapping(mapping):
    for paragraph, pair in mapping.items():
        for key, val in pair.items():
            content = paragraph.content
            print(f"key: {content[key[0]:key[1]]}, value: {content[val[0]:val[1]]}")

class Entity:
    instances = []

    def __init__(self):
        self.mentions: List[Any] = []
        self.alias: List[str] = []
        self.events = []
        self.representive = None
        Entity.instances.append(self)

    def add_mention(self, mention):
        mention.set_referent(self)
        self.alias.append(mention.reify())
        self.mentions.append(mention)

    def get_representative(self):
        return Counter(self.alias).most_common(1)[0][0]
    @staticmethod
    def get_instances():
        reps = []
        for entity in Entity.instances:
            reps.append(entity.get_representative())
        return reps
    @staticmethod
    def maybe_find_referent(mention, mode="strict"):
        if mode not in ["strict", "inclusion"]:
            raise NotImplementedError
        #criterion = (lambda a, b: a == b) if mode == "strict" else partial(is_included_in, portion=0.6)
        mention_criterion = (lambda a, b: a == b) if mode == "strict" else partial(mention_has_intersection, portion=0.5)
        votes: List[int] = [0] * len(Entity.instances)
        for idx, entity in enumerate(Entity.instances):
            for other in entity.mentions:
                if mention.paragraph == other.paragraph and \
                    mention.span == other.span:
                        #criterion(mention.span, other.span):
                    return entity
                if mention_criterion(mention.reify(), other.reify()) and mention.reify() not in DEIXIS:
                    votes[idx] += 1

        votes = np.array(votes)
        if votes.size > 0 and not np.all(votes == 0):
            reps = Entity.get_instances()
            entity_idx = np.argmax(votes)
            print(f"mention: {mention.reify()}; reps: {reps} resolves to {reps[entity_idx]}")
            entity = Entity.instances[entity_idx]
            return entity

        return None

    @staticmethod
    def maybe_find_referent_for_mentions(mentions, mode="strict"):
        if mode not in ["strict", "inclusion"]:
            raise NotImplementedError
        criterion = (lambda a, b: a == b)# if mode == "strict" else partial(is_included_in, portion=0.66)
        votes: List[int] = [0] * len(Entity.instances)
        for idx, entity in enumerate(Entity.instances):
            for mention in mentions:
                for other in entity.mentions:
                    if mention.paragraph == other.paragraph and \
                            criterion(mention.span, other.span):
                        return entity
                    if mention.reify() == other.reify() and mention.reify() not in DEIXIS:
                        votes[idx] += 1

        votes = np.array(votes)
        if len(votes) > 0 and not np.all(votes == 0):
            entity_idx = np.argmax(votes)
            entity = Entity.instances[entity_idx]
            return entity

        return None


class Paragraph:
    def __init__(self,
                 content,
                 offset=None):
        self.offset = offset
        self.content = content


class Mention:
    instances = []
    def __init__(self,
                 paragraph: Paragraph,
                 span: Tuple[int, int]):

        self.paragraph = paragraph
        self.span = span
        self.referent = None
        Mention.instances.append(self)

    def reify(self):
        return "".join(self.paragraph.content[self.span[0]:self.span[1]])

    def maybe_resolve_referent(self):
        self.set_referent(Entity.maybe_find_referent(self, mode="inclusion"))

    def set_referent(self, referent: Union[Entity, None]):
        if referent is not None:
            self.referent = referent

    def __str__(self):
        return self.reify()

    def __repr__(self):
        return self.reify()


class Event:
    def __init__(self,
                 predicate,
                 agent=None,
                 patient=None,
                 other=None,
                 tmp=None,
                 adv=None):
        self.predicate = predicate
        self.agent = agent
        self.patient = patient
        self.other = other
        self.tmp = tmp
        self.adv = adv

    @classmethod
    def parse(cls,
              paragraph: Paragraph,
              triple: Dict[str, Tuple[int, int]]) -> 'Event':
        A0, A1, A2, tmp, adv = None, None, None, None, None
        if "Pred" in triple:
            predicate = Mention(paragraph, triple["Pred"])
        if "A0" in triple:
            A0 = Mention(paragraph, triple["A0"])
            A0.maybe_resolve_referent()
        else:
            try:
                subject = SUBJECTS[paragraph][tuple(triple["Pred"])]
                A0 = Mention(paragraph, subject)
                A0.maybe_resolve_referent()
            except KeyError:
                pass
        if "A1" in triple:
            A1 = Mention(paragraph, triple["A1"])
            A1.maybe_resolve_referent()
        else:
            try:
                #print(paragraph.content[triple["Pred"][0]:triple["Pred"][1]])
                dobj = OBJECTS[paragraph][tuple(triple["Pred"])]
                A1 = Mention(paragraph, dobj)
                A1.maybe_resolve_referent()
            except KeyError:
                pass
        if "A2" in triple:
            A2 = Mention(paragraph, triple["A2"])
            A2.maybe_resolve_referent()
        if "TMP" in triple:
            tmp = Mention(paragraph, triple["TMP"])
        if "ADV" in triple:
            adv = Mention(paragraph, triple["ADV"])

        return cls(predicate, agent=A0, patient=A1, other=A2, tmp=tmp, adv=adv)

    def __str__(self):
        return "pred: {}, A0: {}, A1: {}, A2: {}, TMP: {}, ADV: {}". \
            format(self.predicate, self.agent, self.patient,
                   self.other, self.tmp, self.adv)


def construct_digraph(events: List[Event]):
    core_graph = Digraph(graph_attr={
        # 'nodesep': '1.0',
        # 'ranksep': '1.0',
        'rankdir': 'LR'
    })
    peripheral_graph = Digraph(graph_attr={
        # 'nodesep': '1.0',
        # 'ranksep': '1.0',
        'rankdir': 'LR'
    })
    interval = 10
    core_events = []
    peripheral_events = []
    for event in events:
        if event.agent is None or event.patient is None:
            continue
        if event.agent.referent is None and event.patient.referent is None:
            peripheral_events.append(event)
        else:
            core_events.append(event)

    for events, graph in zip([core_events, peripheral_events], [core_graph, peripheral_graph]):
        for event in events:
            agent = None
            if event.agent is not None:
                #agent = event.agent.reify()
                if event.agent.referent is not None:
                    agent = event.agent.referent.representative
                else:
                    agent = event.agent.reify()
                graph.node(line_break(agent, interval), line_break(agent, interval))
            #else:
                #agent = event.patient.reify() + "dummy_agent"
                #graph.node(agent, " ")

            patient = None
            if event.patient is not None:
                #patient = event.patient.reify()
                if event.patient.referent is not None:
                    patient = event.patient.referent.representative
                else:
                    #continue
                    patient = event.patient.reify()
                graph.node(line_break(patient, interval), line_break(patient, interval))
            #else:
                #patient = event.agent.reify() + "dummy_patient"
                #graph.node(patient, " ")
            print(f"A0: {event.agent.reify()}, predicate: {event.predicate.reify()}, A1: {event.patient.reify()}")
            if agent and patient:
                #agent = line_break(agent, interval)
                #patient = line_break(patient, interval)
                graph.edge(line_break(agent, interval),
                           line_break(patient, interval),
                           label=event.predicate.reify())

    return core_graph, peripheral_graph


def span_dist(span1, span2):
    return min(np.abs(span1[-1] - span2[0]), np.abs(span1[0] - span2[-1]))


def process_semgrex_output(paragraph, infos, semgrexes, mode='nsubj'):
    if mode not in ['nsubj', 'dobj']:
        raise NotImplementedError
    
    if mode == "nsubj":
        mapping = SUBJECTS
        match_key = "$subject"
    elif mode == "dobj":
        mapping = OBJECTS
        match_key = "$object"
    

    for info_sent, sent in zip(infos["sentences"], semgrexes['sentences']):
        tokens = info_sent["tokens"]
        for (_, match) in sent.items():
            if isinstance(match, int):
                continue
            token = tokens[match["begin"]]
            span = get_span(token)
            val_span = get_span(tokens[match[match_key]["begin"]])
            try:
                p_map = mapping[paragraph]
                try:
                    old_val_span = p_map[span]
                    p_map[span] = old_val_span \
                        if span_dist(old_val_span, span) < span_dist(val_span, span) \
                        else val_span
                except:
                    p_map[span] = val_span

            except KeyError:
                mapping[paragraph] = {}
                mapping[paragraph][span] = val_span


def semgrex_query_factory(pattern):
    opencc = OpenCC('t2s')
    def func(text):
        simp_text = opencc.convert(text)
        query_string = {"pattern": pattern,
                        "properties": {
                            "annotators": "tokenize,ssplit,pos,ner,depparse,openie",
                            "date": "2019-08-17T21:03:47"
                        }
                        }
        response = requests.post("http://localhost:9001",
                                    params=query_string,
                                 data=dict(data=simp_text))
        infos = response.json()
        response = requests.post("http://localhost:9001/semgrex",
                                    params=query_string,
                                    data=dict(data=simp_text))
        semgrex = response.json()
        return infos, semgrex

    return func

def process_semgrex_triple_output(infos, semgrexes):

    arg_keys = ["$subject", "$object"]
    role_keys = ["A0", "A1"]

    for info_sent, sent in zip(infos["sentences"], semgrexes['sentences']):
        tokens = info_sent["tokens"]
        event = {}
        for (_, match) in sent.items():
            if isinstance(match, int):
                continue
            token = tokens[match["begin"]]
            span = get_span(token)
            event["Pred"] = span
            for role_key, arg_key in zip(role_keys, arg_keys):
                val_span = get_span(tokens[match[arg_key]["begin"]])
                event[role_key] = val_span
        events.append(event)
    
    return events

def process_infos(infos, paragraph):
    for sentence in infos['sentences']:
        tokens = sentence["tokens"]
        for token in tokens:
            if token['ner'] == "PERSON":
                print(token['ner'])
                span = get_span(token)
                mention = Mention(paragraph, span)
                entity = Entity.maybe_find_referent(mention, mode="inclusion")
                if entity is None:
                    entity = Entity()
                entity.add_mention(mention)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input json file")
    parser.add_argument("--output", help="output png file")
    parser.add_argument("--mode", choices=["strict", "inclusion"], help="mode for resolving referents")
    args = parser.parse_args()
    with open(args.input) as fp:
        raw_paragraphs = json.load(fp)

    offset = 0
    seg = Wordseg(batch_size=64, device="cpu", embedding="elmo", elmo_use_cuda=False, mode="TW")
    #parser = benepar.Parser("benepar_zh")

    events: List[Event] = []
    #stanfordnlp.download('zh')
    segmented_paragraphs = seg.cut([para["text"] for para in raw_paragraphs])
    for raw_paragraph, segmented_paragraph in zip(raw_paragraphs, segmented_paragraphs):
        response = requests.post("http://140.112.21.83:8000/predict",
                                 json={"document": raw_paragraph["text"]})

        nsubj_pattern = "{pos:/V.*/} [[<< ({pos:/V.*/} >nsubj {pos:/N.*/}=subject)\& ! >> nsubj {}] | >nsubj {pos:/N.*/}=subject ]"
        #nsubj_pattern = "{pos:/V.*/} [[<<ccomp ({pos:/V.*/} >nsubj {pos:/N.*/;ner:PERSON}=subject)\& ! >> nsubj {}] |>nsubj {pos:/N.*/;ner:PERSON}=subject ]  ?[ >advmod {}=adverb] ? [>dobj ({pos:/N.*/;ner:PERSON}=object ? >nmod {pos:/N.*/}=nmod)]"
        dobj_pattern = "{pos:/V.*/} >dobj {}=object"
        #triple_pattern = "{pos:/V.*/} [[<<ccomp ({pos:/V.*/} >nsubj {pos:/N.*/;ner:PERSON}=subject)\& ! >> nsubj {}] |>nsubj {pos:/N.*/;ner:PERSON}=subject ] [>dobj {pos:/N.*/;ner:PERSON}=object]"
        #query_triple = semgrex_query_factory(triple_pattern)
        #yet_another_triple_pattern = "{pos:/V.*/} [[<<conj ({pos:/V.*/} >nsubj {pos:/N.*/;ner:PERSON}=subject)\& ! >> nsubj {}] |>nsubj {pos:/N.*/;ner:PERSON}=subject ]  [>dobj {ner:PERSON}=obj  |  >dobj  ({pos:/N.*/}=obj >/.*/ {ner:PERSON}=person)]"
        query_nsubj = semgrex_query_factory(nsubj_pattern)
        query_dobj = semgrex_query_factory(dobj_pattern)


        text = raw_paragraph["text"]
        infos, dobjs = query_dobj(text)
        infos, nsubjs = query_nsubj(text)
        #infos, semgrex_triples = query_triple(text)
        clusters = response.json()["clusters"]
        paragraph = Paragraph(text, offset)
        process_infos(infos, paragraph)

        process_semgrex_output(paragraph, infos, nsubjs, mode='nsubj')
        process_semgrex_output(paragraph, infos, dobjs, mode='dobj')
        #syntactic_triples = process_semgrex_triple_output(infos, semgrex_triples)
        #print_mapping(SUBJECTS)
        #print_mapping(OBJECTS)
        #print(paragraph.content)
        offset += len(paragraph.content)
        for raw_mentions in clusters:
            mentions = []
            for raw_mention in raw_mentions:
                raw_mention[1] += 1 # fix span to slice
                mention = Mention(paragraph, raw_mention)
                mentions.append(mention)
                if mention.reify() == "母親陳賀":
                    print(f"mentions: {[mention.reify() for mention in mentions]}")
            entity = Entity.maybe_find_referent_for_mentions(mentions, mode="inclusion")
            if entity is not None:
                for mention in mentions:
                    entity.add_mention(mention)

        for triple in raw_paragraph["args"]:
            event = Event.parse(paragraph, triple)
            events.append(event)


        # for event in events:
        #     for participant in [event.agent, event.patient, event.other]:
        #         if participant is not None:
        #             if participant.referent is not None:
        #                 participant.referent.events.append(event)

    for entity in Entity.instances:
        c = Counter(entity.alias)
        representative = c.most_common(1)[0][0]
        entity.representative = representative
    for mention in Mention.instances:
        mention.maybe_resolve_referent()
        if mention.referent is not None:
            print(f"mention: {mention.reify()}, referent: {mention.referent.representative}")
    core_graph, peripheral_graph = construct_digraph(events)
    core_graph.render(args.output + '_core')
    peripheral_graph.render(args.output + '_peripheral')
