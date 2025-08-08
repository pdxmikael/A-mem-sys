import os
import json
from uuid import uuid4
from typing import Any, Dict, List, Tuple

from agentic_memory.memory_system import AgenticMemorySystem
from agentic_memory.retrievers import ChromaRetriever

import dotenv
dotenv.load_dotenv()


def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_memories(ms: AgenticMemorySystem, note: str = "Current Memories") -> None:
    print_header(note)
    for mid, m in ms.memories.items():
        print(json.dumps({
            "id": mid,
            "content": m.content,
            "context": m.context,
            "keywords": m.keywords,
            "tags": m.tags,
            "category": m.category,
            "timestamp": m.timestamp,
            "retrieval_count": m.retrieval_count,
            "links": m.links,
        }, ensure_ascii=False, indent=2))
    if not ms.memories:
        print("<no memories>")


def format_results(results: Dict[str, Any]) -> List[Tuple[str, float]]:
    """Flatten Chroma query results -> List[(id, distance)]"""
    if not results or not results.get("ids"):
        return []
    ids = results.get("ids", [[]])[0]
    dists = results.get("distances", [[]])[0]
    out: List[Tuple[str, float]] = []
    for i, mid in enumerate(ids):
        if i < len(dists):
            try:
                out.append((mid, float(dists[i])))
            except Exception:
                out.append((mid, 0.0))
    return out


def compare_and_print(ms_with: AgenticMemorySystem, ms_without: AgenticMemorySystem, query: str, k: int = 5, title_suffix: str = "") -> None:
    print_header(f"Distance comparison (lower is closer) -> '{query}' {title_suffix}")

    # Run raw retriever queries to ensure we compare the same metric
    where = {"session_id": ms_with.session_id}
    res_yes = ms_with.retriever.search(query, k=k, where=where)
    res_no = ms_without.retriever.search(query, k=k, where=where)

    flat_yes = format_results(res_yes)
    flat_no = format_results(res_no)

    # Build maps for easy comparison
    map_yes = {mid: dist for mid, dist in flat_yes}
    map_no = {mid: dist for mid, dist in flat_no}

    # Union of IDs in both results
    all_ids: List[str] = []
    seen = set()
    for mid, _ in flat_no + flat_yes:
        if mid not in seen:
            seen.add(mid)
            all_ids.append(mid)

    if not all_ids:
        print("<no results>")
        return

    # Print rows
    rows = []
    for mid in all_ids:
        d_no = map_no.get(mid, float("inf"))
        d_yes = map_yes.get(mid, float("inf"))
        delta = d_no - d_yes if (d_no != float("inf") and d_yes != float("inf")) else None
        rows.append({
            "id": mid,
            "distance_without": d_no,
            "distance_with": d_yes,
            "delta": delta,
        })

    # Sort by with-shaper distance to show best matches under shaping
    rows.sort(key=lambda r: r["distance_with"])  # ascending: lower is better

    for r in rows:
        print(json.dumps(r, ensure_ascii=False, indent=2))


def main() -> None:
    # Create a single session that both systems will use, so we only add data once
    session_id = f"demo-shape-{uuid4()}"
    backend = os.getenv("DEFAULT_LLM_BACKEND", "openai")
    model = os.getenv("DEFAULT_LLM_MODEL", "gpt-5-mini")
    persist_dir = "./memory_db"

    print_header("Initializing AgenticMemorySystem (without and with QueryShaper)")
    print(json.dumps({
        "session_id": session_id,
        "llm_backend": backend,
        "llm_model": model,
    }, indent=2))

    # System WITHOUT QueryShaper
    ms_no = AgenticMemorySystem(
        session_id=session_id,
        llm_backend=backend,
        llm_model=model,
        api_key=os.getenv("OPENAI_API_KEY"),
        persist_directory=persist_dir,
    )
    # Ensure retriever has shaping disabled
    ms_no.retriever = ChromaRetriever(
        collection_name="memories",
        model_name=ms_no.model_name,
        persist_directory=persist_dir,
        use_query_shaper=False,
    )

    try:
        # Add memories ONCE using the no-shaper system
        print_header("Adding 5 memories (baseline, no shaper)")

        # 1) Arctic field log
        m1 = ms_no.add_note(
            "Recovered Echo-7 beacon near Black Ridge; repaired snowmobile fuel line.",
            context=(
                """
                The wind rolled down from the serrated teeth of Black Ridge as a pale wall, scouring the
                camp in a steady hiss that sifted snow into every zipper and seam. We spent the pre-dawn
                hour at Camp Borealis coaxing the snowmobile back to life, fingers stiff inside mitts as
                I traced the fuel line with a headlamp. The split was hair-thin but long enough to choke
                the engine under load; a spare length of hose and a double clamp did the trick. By the
                time the eastern sky bruised purple, the GPS console pulsed with the faint, intermittent
                ping of the Echo-7 beacon — our lost scout drone — showing somewhere along a contour just
                southwest of the Black Ridge cornice.

                The snow carried a story of its own. Fresh tracks — wide, deliberate, the pads stitched
                with the faintest starburst of hair — cut across the drainage that led toward the ridge.
                Polar bear, adult, moving with the wind. I marked the bearing and kept my voice low over
                the radio as I relayed the note to base: “Bear sign near Marker J-12, moving west.” The
                air tasted metallic in my balaclava, and every breath laced the goggles with a soft halo.

                We advanced in short legs, the repaired sled cresting shallow drifts like a seal through
                foam. At a break in the snow’s crust a glint flashed: not ice, not mica, but the bronze
                belly of Echo-7, overturned and half-eaten by rime. Its starboard rotor was a thicket of
                shattered carbon, the antenna snapped, but the beacon still tugged at the receiver with a
                dogged, slow heartbeat. I freed it from the drift, read the last telemetry burst, and
                clipped the chassis to the cargo rack. The horizon lowered as the storm front shouldered
                closer, the ridge sharpening like a saw. We turned for camp, following our own ribs of
                compressed snow back toward the scatter of crates and the orange dome of the weather tent.
                As we pulled in, I could hear the anemometer sing like a wire in the wind, a high steady
                note that meant: not long now. I logged the salvage at 09:42 and marked Black Ridge and
                Camp Borealis on the incident sheet, noting the bear track and the Echo-7 recovery.
                """
            ),
            keywords=["Echo-7", "Black Ridge", "Camp Borealis", "snowmobile", "beacon"],
            tags=["arctic", "field_log"],
            category="Expedition",
        )

        # 2) Astronomer's Guild heist
        m2 = ms_no.add_note(
            "Used octagonal copper key to enter Room Helios; secured Seraphim Map fragment at the Astronomer's Guild.",
            context=(
                """
                Valeris sleeps uneasily under its hundred domes, the observatories blinking like patient
                eyes across the ridge. The Astronomer’s Guild keeps its secrets in a ceramic hive of rooms
                named for the sun’s moods — Meridian, Duskfall, Helios — each with its own peculiar lock.
                Tonight, the sky was a lacquered bowl and the guard’s boots echoed, predictable as a
                metronome, along the mezzanine. I timed my breath with his turn. When the lantern halo swept
                the balustrade, I slid from the shadow beneath the Orrery of Saints and slipped down the
                spiral, a flake among flurries.

                Room Helios wears an iris-lock like a flower clenched against weather. The octagonal copper
                key, warm from the press of my palm through the cloth, ticked each petal into motion. Eight
                clicks like distant bells, a low sigh of gears, and the aperture dilated to spill a wedge of
                honey light across the floor. Inside, the Seraphim Map fragment waited in a frame of black
                walnut, its vellum scored with star-lattice and inked in a dialect that never learned to
                bow. The chart’s southern rim carries the faint watermark of the Maelion Scriptorium: the
                last proof we need that the Guild hoarded more than star-weather.

                I cradled the fragment with paper gloves and rolled it into the bone tube I’d hidden in my
                sash. Behind me, the orrery ticked, brass planets nodding approval as they swam along their
                orbits. A clock coughed twice from the archive hall — the guard’s cue for shift change — and
                I eased the iris shut, resetting the petals with a turn that left no trace of brass on brass.
                Outside, the river bit the air and carried it toward the docks. I crossed the Quadrant of
                Measures, counting flagstones, and tucked myself into the seam of two vaulted buttresses as
                the lantern passed again. When the bell tower shook out the hour, I was already another face
                among market shadows, copper key quiet in my pocket, Room Helios a memory cooling to silence.
                """
            ),
            keywords=["octagonal copper key", "Room Helios", "Seraphim Map", "Astronomer’s Guild"],
            tags=["heist", "valeris"],
            category="Caper",
        )

        # 3) Spaceport run
        m3 = ms_no.add_note(
            "Rerouted Kestrel-9 through Service Spine 3 to bypass CL-12; delivered iridium capacitors to Lysa Vorn at Pier Delta.",
            context=(
                """
                Pier Delta glows a sickly turquoise where the coolant vents bleed into fog, painting the hulls
                in a bruise. Kestrel-9 had been sulking all afternoon, a hairline short in the dorsal conduit
                throwing phantom alerts across the nav. The customs drone — CL-12, the chatty one with a scuff
                on its speaker grille — hovered like a bored gull above the checkpoint, chirping for random
                inspections. The manifest said “machine parts,” but the crate smelled faintly of ozone and a
                miner’s optimism — iridium capacitors, the kind that turn cutters into whispers.

                Lysa Vorn was late, which meant the window would be small. I dumped power from the fore lights
                and swung the service panel on the bulkhead between bays seven and eight. The Service Spine 3
                is not a passage so much as an apology: ducts, cables, a knee-bruiser of a ladder, and heat
                like a big cat’s breath. CL-12 hummed toward a new target as I slipped the crate along the
                spine, counting meters in the dim: four, eight, twelve — turn right where the insulation peels.
                A maintenance engineer cursed somewhere above me; a ship’s belly echoed the complaint.

                We came up under a loose grating near the recycler feed. Lysa’s coat flashed tin buttons and a
                smile sharp enough to cut a seal. “You’re a poem,” she said, and hefted the crate with a grunt
                that made poetry look like work. I rerouted the Kestrel-9’s power to starboard, dipped thrusters
                to slide past CL-12’s patrol cone, and let the fog finish our sentence. The dockmaster’s board
                blinked green. By the time CL-12 spun back to scold the darkness, we were already a smear of
                condensation on its lenses and a payment ping in my pocket.
                """
            ),
            keywords=["Kestrel-9", "CL-12", "Service Spine 3", "Lysa Vorn", "iridium capacitors", "Pier Delta"],
            tags=["smuggling", "spaceport"],
            category="SciFi",
        )

        # 4) Village remedy
        m4 = ms_no.add_note(
            "Boiled kingsfoil at Old Ashen Well to treat Hobb’s redblight in Brindleford.",
            context=(
                """
                Brindleford’s lane is a ribbon of mud plaited with straw, and the mill’s wheel turns with a
                tired slosh that seems to sigh on every rotation. The redblight took Hobb first — a rash like
                autumn maples, fever like a kiln — and then it made a map of his cottage walls with coughs. The
                almanac says the Witchfen keeps a memory of every illness, and the edge of the fen keeps a plant
                for every memory, if you pay the right price to the midges and the mud. I took a kettle, a hoop
                of twine, and a satchel with more hope than sense, and went to bargain.

                Kingsfoil grows where the fen remembers fire. By the Old Ashen Well the birches wear soot like
                jewelry and the water tastes a little of thunder. I plucked sprigs until resin turned my palms
                green and the air smelled like mended hearts. Back in Brindleford the midwife tapped the kettle
                with a spoon and called the boil by its proper name, coaxing the steam to carry the herb where it
                must. Hobb’s breath came easier by inches, then by spans; the red receded like a tide that had
                never meant to drown the shore. We left a ribbon on the well rope for thanks and a coin in the
                bucket for luck that looks like gratitude.
                """
            ),
            keywords=["Brindleford", "kingsfoil", "Old Ashen Well", "redblight", "Hobb"],
            tags=["remedy", "village"],
            category="Fantasy",
        )

        # 5) Undersea station
        m5 = ms_no.add_note(
            "Patched helium-argon leak and recovered Aphid-2 near Sable Chimneys; Blue Lantern passed systems check F3-Delta.",
            context=(
                """
                Pelagia Deep-4 hangs under the black skin of the southern trench, its corridors humming in a key
                you feel more than hear. The submersible Blue Lantern sulked in Dock Two, sulking as machines do:
                a fault code F3-Delta winking like a lazy eye. Outside the viewport the hydrothermal field called
                Sable Chimneys exhaled its pale pillars, venting in curtains that folded and unfolded in slow
                applause. Aphid-2 — a probe with pretensions — had gone missing at the edge of the plume, and the
                comm buoy Sigma had been complaining in clipped bursts about pressure drift and nonsense.

                We bled the lines, checked the regulators, and found the leak by smell more than sound — a thin
                helium-argon mix that made our voices comic in the mask radios, like the station had decided to
                laugh at its own joke. Patch laid and cured, Blue Lantern leveled her tone and let me guide her
                into the dark. The beam cut a cone through snowing silt, and there, cribbed against a basalt lip,
                Aphid-2 blinked its wounded green. A manipulator arm scooped it tenderly as a cat rescues a toy,
                and we swung back on a bearing that put the station’s heartbeat back in our bones. Back in Dock
                Two the diagnostics sang a clean line; F3-Delta turned to a smile. I logged the recovery, the
                patch, and the complaint to Buoy Sigma, which responded, at last, with what I chose to hear as
                approval.
                """
            ),
            keywords=["Pelagia Deep-4", "Blue Lantern", "Sable Chimneys", "Aphid-2", "F3-Delta", "Buoy Sigma"],
            tags=["undersea", "research"],
            category="Ocean",
        )

        print_memories(ms_no, "After adding memories (baseline)")

        # Create a second system WITH QueryShaper that loads the same session's data
        ms_yes = AgenticMemorySystem(
            session_id=session_id,
            llm_backend=backend,
            llm_model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            persist_directory=persist_dir,
        )
        ms_yes.retriever = ChromaRetriever(
            collection_name="memories",
            model_name=ms_yes.model_name,
            persist_directory=persist_dir,
            use_query_shaper=True,
        )

        # Roleplay next-action queries
        query1 = (
            "I am the polar guide. I cinch my hood, fire up the repaired snowmobile, and ride for Black Ridge "
            "to recover the Echo-7 beacon before the whiteout sweeps Camp Borealis. I’ll trace the bear tracks "
            "and haul the drone back to base."
        )
        query2 = (
            "Staying in character as the cat burglar, I slide the octagonal copper key into the Astronomer’s "
            "Guild iris and step into Room Helios to snatch the Seraphim Map fragment before the guard rounds "
            "the mezzanine."
        )

        # Show standard printed results as before (optional):
        print_header("Regular search results (no shaper)")
        print(json.dumps(ms_no.search(query1, k=5), indent=2))
        print_header("Regular search results (with shaper)")
        print(json.dumps(ms_yes.search(query1, k=5), indent=2))

        print_header("Agentic search results (no shaper)")
        print(json.dumps(ms_no.search_agentic(query2, k=5), indent=2))
        print_header("Agentic search results (with shaper)")
        print(json.dumps(ms_yes.search_agentic(query2, k=5), indent=2))

        # Distance comparisons (retriever raw distances) for both queries
        compare_and_print(ms_with=ms_yes, ms_without=ms_no, query=query1, k=5, title_suffix="(Regular)")
        compare_and_print(ms_with=ms_yes, ms_without=ms_no, query=query2, k=5, title_suffix="(Agentic base retrieval)")

    finally:
        # Clean up the session data to avoid polluting persistent store
        try:
            print_header("Cleaning up session data")
            deleted = ms_no.delete_all_by_session(session_id)
            print(json.dumps({"session_id": session_id, "deleted": deleted}, indent=2))
        except Exception as e:
            print(f"Cleanup error: {e}")


if __name__ == "__main__":
    main()
