"""A small, original 'wiki-style' corpus + chunking — offline, no downloads.

Encyclopedia-style articles written from scratch on neutral topics, so the whole
system runs with no external data. Each article is split into overlapping chunks
(the unit that gets indexed and retrieved), mirroring how a real wiki dump is
prepared for RAG. Point ``load_corpus`` at real articles without changing the
index, reranker, or search pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

# neutral, original encyclopedia-style entries (title -> body)
WIKI: dict[str, str] = {
    "Photosynthesis": (
        "Photosynthesis is the process by which green plants, algae, and some "
        "bacteria convert light energy into chemical energy. Chlorophyll in the "
        "chloroplasts absorbs sunlight, and water and carbon dioxide are turned "
        "into glucose and oxygen. The oxygen is released into the air, and the "
        "glucose stores energy the organism uses to grow."),
    "Volcano": (
        "A volcano is a rupture in the crust of a planet that lets hot magma, "
        "ash, and gases escape from a chamber below the surface. Eruptions can "
        "be explosive or effusive. Over time, cooled lava builds cones and "
        "shield mountains. Volcanoes often form near the boundaries of tectonic "
        "plates where the crust is weak."),
    "Photography": (
        "Photography is the art and practice of creating images by recording "
        "light on a sensor or film. A lens focuses light through an aperture, "
        "and the shutter controls how long the sensor is exposed. Adjusting "
        "aperture, shutter speed, and sensitivity changes exposure and depth of "
        "field. Digital cameras store the image as a file."),
    "Honeybee": (
        "The honeybee is a flying insect known for producing honey and for "
        "pollinating flowers. A colony has a single queen, many female workers, "
        "and seasonal male drones. Workers collect nectar and pollen, communicate "
        "the location of food through a waggle dance, and build wax combs to "
        "store honey and raise young."),
    "Glacier": (
        "A glacier is a large, persistent body of ice that forms where snow "
        "accumulates faster than it melts and slowly compresses into ice. Under "
        "its own weight the ice flows downhill, carving valleys and carrying rock "
        "debris. When a glacier reaches the sea it can break apart, or calve, into "
        "icebergs. Glaciers store much of the world's fresh water."),
    "Tides": (
        "Tides are the periodic rise and fall of sea levels caused mainly by the "
        "gravitational pull of the Moon and the Sun on the oceans. As the Earth "
        "rotates, most coasts experience two high tides and two low tides each "
        "day. The difference in height between high and low water is the tidal "
        "range, which is largest during spring tides."),
}


@dataclass
class Chunk:
    id: str
    title: str
    text: str


def chunk_text(text: str, size: int = 24, overlap: int = 8) -> list[str]:
    """Split into overlapping word windows (the retrieval unit)."""
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if window:
            out.append(" ".join(window))
        if start + size >= len(words):
            break
    return out


def load_corpus(articles: dict[str, str] | None = None,
                size: int = 24, overlap: int = 8) -> list[Chunk]:
    articles = articles or WIKI
    chunks: list[Chunk] = []
    for title, body in articles.items():
        for i, ct in enumerate(chunk_text(body, size=size, overlap=overlap)):
            chunks.append(Chunk(id=f"{title}#{i}", title=title, text=ct))
    return chunks
