from typing import NamedTuple, Optional, List, Callable, Tuple, Iterable
import xml.etree.cElementTree as ET
from xml.dom import minidom

pos_map = {
    # U-POS
    "NOUN": "n",
    "VERB": "v",
    "ADJ": "a",
    "ADV": "r",
    "PROPN": "n",
    # PEN
    "AFX": "a",
    "JJ": "a",
    "JJR": "a",
    "JJS": "a",
    "MD": "v",
    "NN": "n",
    "NNP": "n",
    "NNPS": "n",
    "NNS": "n",
    "RB": "r",
    "RP": "r",
    "RBR": "r",
    "RBS": "r",
    "VB": "v",
    "VBD": "v",
    "VBG": "v",
    "VBN": "v",
    "VBP": "v",
    "VBZ": "v",
    "WRB": "r",
}


class AnnotatedToken(NamedTuple):
    text: str
    pos: Optional[str] = None
    lemma: Optional[str] = None


class WSDInstance(NamedTuple):
    annotated_token: AnnotatedToken
    labels: Optional[List[str]]
    instance_id: Optional[str]


def read_from_raganato(
    xml_path: str,
    key_path: Optional[str] = None,
    instance_transform: Optional[Callable[[WSDInstance], WSDInstance]] = None,
) -> Iterable[Tuple[str, str, List[WSDInstance]]]:
    def read_by_text_iter(xml_path: str):

        it = ET.iterparse(xml_path, events=("start", "end"))
        _, root = next(it)

        for event, elem in it:
            if event == "end" and elem.tag == "text":
                document_id = elem.attrib["id"]
                for sentence in elem:
                    sentence_id = sentence.attrib["id"]
                    for word in sentence:
                        yield document_id, sentence_id, word

            root.clear()

    mapping = {}

    if key_path is not None:
        try:
            with open(key_path) as f:
                for line in f:
                    line = line.strip()
                    wsd_instance, *labels = line.split(" ")
                    mapping[wsd_instance] = labels
        except Exception:
            pass

    last_seen_document_id = None
    last_seen_sentence_id = None

    for document_id, sentence_id, element in read_by_text_iter(xml_path):

        if last_seen_sentence_id != sentence_id:

            if last_seen_sentence_id is not None:
                yield last_seen_document_id, last_seen_sentence_id, sentence

            sentence = []
            last_seen_document_id = document_id
            last_seen_sentence_id = sentence_id

        annotated_token = AnnotatedToken(
            text=element.text,
            pos=element.attrib.get("pos", None),
            lemma=element.attrib.get("lemma", None),
        )

        wsd_instance = WSDInstance(
            annotated_token=annotated_token,
            labels=None
            if element.tag == "wf" or element.attrib["id"] not in mapping
            else mapping[element.attrib["id"]],
            instance_id=None if element.tag == "wf" else element.attrib["id"],
        )

        if instance_transform is not None:
            wsd_instance = instance_transform(wsd_instance)

        sentence.append(wsd_instance)

    yield last_seen_document_id, last_seen_sentence_id, sentence


def expand_raganato_path(path: str) -> Tuple[str, str]:
    path = path.replace(".data.xml", "").replace(".gold.key.txt", "")
    return f"{path}.data.xml", f"{path}.gold.key.txt"
