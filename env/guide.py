"""Pokemon Blue Game Guide — rule-based progression tracker.

This is the second "AI" in the two-agent system.  It is entirely
deterministic and requires no training — it simply reads the current
game state, works out which milestone has been reached, and tells the
PPO learner:

  1. How far through the game it is (normalised progress scalar)
  2. Which map it should be heading toward next (target map hint)
  3. A one-time bonus reward every time a new milestone is unlocked
  4. A human-readable hint describing what to do at the current stage

Milestone state intentionally persists across episodes (like
tile_visit_count / visited_maps) so the guide always points toward
genuinely new territory rather than resetting its knowledge each time.

Walkthrough source: bulbapedia.bulbagarden.net/wiki/
    Walkthrough:Pokémon_Red_and_Blue/Part_1  …/Part_17
"""

# ---------------------------------------------------------------------------
# Map ID constants  (decimal values stored at 0xD35E in WRAM)
# ---------------------------------------------------------------------------

MAP_PALLET_TOWN      = 0
MAP_VIRIDIAN_CITY    = 1
MAP_PEWTER_CITY      = 2
MAP_CERULEAN_CITY    = 3
MAP_LAVENDER_TOWN    = 4
MAP_VERMILION_CITY   = 5
MAP_CELADON_CITY     = 6
MAP_FUCHSIA_CITY     = 7
MAP_CINNABAR_ISLAND  = 8
MAP_INDIGO_PLATEAU   = 9
MAP_SAFFRON_CITY     = 10

MAP_ROUTE_1          = 12
MAP_ROUTE_2          = 13
MAP_ROUTE_3          = 14
MAP_ROUTE_4          = 15
MAP_ROUTE_22         = 33
MAP_ROUTE_24         = 35   # Nugget Bridge / Bill approach
MAP_ROUTE_25         = 36   # Bill's Sea Cottage
MAP_VIRIDIAN_FOREST  = 37
MAP_MT_MOON_1F       = 38
MAP_MT_MOON_B1F      = 39
MAP_MT_MOON_B2F      = 40
MAP_ROCK_TUNNEL_1F   = 41
MAP_POKEMON_TOWER_1F = 43
MAP_SEAFOAM_1F       = 51
MAP_VICTORY_ROAD_1F  = 57
MAP_SS_ANNE          = 64
MAP_SILPH_CO_1F      = 92
MAP_POKEMON_MANSION  = 103
MAP_SAFARI_ZONE      = 107

MAP_VIRIDIAN_GYM     = 113
MAP_PEWTER_GYM       = 114
MAP_CERULEAN_GYM     = 115
MAP_VERMILION_GYM    = 116
MAP_CELADON_GYM      = 117
MAP_FUCHSIA_GYM      = 118
MAP_SAFFRON_GYM      = 119
MAP_CINNABAR_GYM     = 120

MAP_E4_LORELEI       = 121
MAP_E4_BRUNO         = 122
MAP_E4_AGATHA        = 123
MAP_E4_LANCE         = 124
MAP_CHAMPION_ROOM    = 125
MAP_HALL_OF_FAME     = 126
MAP_OAKS_LAB         = 127

# ---------------------------------------------------------------------------
# Badge bitmasks  (0xD356)
# ---------------------------------------------------------------------------

BADGE_BOULDER  = 0x01   # Brock     — Pewter Gym
BADGE_CASCADE  = 0x02   # Misty     — Cerulean Gym
BADGE_THUNDER  = 0x04   # Lt. Surge — Vermilion Gym
BADGE_RAINBOW  = 0x08   # Erika     — Celadon Gym
BADGE_SOUL     = 0x10   # Koga      — Fuchsia Gym
BADGE_MARSH    = 0x20   # Sabrina   — Saffron Gym
BADGE_VOLCANO  = 0x40   # Blaine    — Cinnabar Gym
BADGE_EARTH    = 0x80   # Giovanni  — Viridian Gym

# ---------------------------------------------------------------------------
# Milestone table
#
# Each entry: (name, next_target_map_id, detector_fn)
#
# detector_fn(badge_bitmask: int, visited_maps: set[int]) -> bool
#   Returns True once this milestone has been reached.
#
# Milestones are strictly ordered — the guide advances linearly and only
# checks the detector for milestone N+1 once milestone N is reached.
#
# MILESTONE_HINTS (parallel list) gives a human-readable description of
# what the agent should do to reach the NEXT milestone.
# ---------------------------------------------------------------------------

MILESTONES = [
    # -----------------------------------------------------------------------
    # PHASE 1 — Pallet Town → Boulder Badge
    # -----------------------------------------------------------------------

    # 0 — game starts in Pallet Town
    ("start",
     MAP_VIRIDIAN_CITY,
     lambda b, vm: True),

    # 1 — set foot on Route 1
    ("route_1",
     MAP_VIRIDIAN_CITY,
     lambda b, vm: MAP_ROUTE_1 in vm),

    # 2 — reached Viridian City; must visit Poke Mart, get Oak's Parcel,
    #     deliver it back to Oak, then return north through Viridian Forest
    ("viridian_city",
     MAP_VIRIDIAN_FOREST,
     lambda b, vm: MAP_VIRIDIAN_CITY in vm),

    # 3 — entered Viridian Forest; parcel delivered, old man has moved
    ("viridian_forest",
     MAP_PEWTER_CITY,
     lambda b, vm: MAP_VIRIDIAN_FOREST in vm),

    # 4 — reached Pewter City
    ("pewter_city",
     MAP_PEWTER_GYM,
     lambda b, vm: MAP_PEWTER_CITY in vm),

    # 5 — Boulder Badge (Brock, Lv 12 Geodude / Lv 14 Onix)
    ("badge_boulder",
     MAP_MT_MOON_1F,
     lambda b, vm: bool(b & BADGE_BOULDER)),

    # -----------------------------------------------------------------------
    # PHASE 2 — Boulder Badge → Cascade Badge
    # -----------------------------------------------------------------------

    # 6 — entered Mt. Moon (Route 3 is between here and Pewter)
    ("mt_moon",
     MAP_CERULEAN_CITY,
     lambda b, vm: MAP_MT_MOON_1F in vm),

    # 7 — reached the deepest floor of Mt. Moon; choose Dome or Helix Fossil
    ("mt_moon_b2f",
     MAP_CERULEAN_CITY,
     lambda b, vm: MAP_MT_MOON_B2F in vm),

    # 8 — reached Cerulean City
    ("cerulean_city",
     MAP_CERULEAN_GYM,
     lambda b, vm: MAP_CERULEAN_CITY in vm),

    # 9 — Cascade Badge (Misty, Lv 18 Staryu / Lv 21 Starmie)
    ("badge_cascade",
     MAP_ROUTE_24,
     lambda b, vm: bool(b & BADGE_CASCADE)),

    # -----------------------------------------------------------------------
    # PHASE 3 — Cascade Badge → Thunder Badge
    # -----------------------------------------------------------------------

    # 10 — crossed Nugget Bridge (Route 24); heading to Bill
    ("nugget_bridge",
     MAP_ROUTE_25,
     lambda b, vm: MAP_ROUTE_24 in vm),

    # 11 — reached Bill's Sea Cottage (Route 25); get S.S. Ticket
    ("bills_cottage",
     MAP_VERMILION_CITY,
     lambda b, vm: MAP_ROUTE_25 in vm),

    # 12 — reached Vermilion City; board S.S. Anne for HM Cut
    ("vermilion_city",
     MAP_SS_ANNE,
     lambda b, vm: MAP_VERMILION_CITY in vm),

    # 13 — boarded S.S. Anne; help the Captain to receive HM01 Cut
    ("ss_anne",
     MAP_VERMILION_GYM,
     lambda b, vm: MAP_SS_ANNE in vm),

    # 14 — Thunder Badge (Lt. Surge, Lv 21 Voltorb / Lv 18 Pikachu / Lv 24 Raichu)
    ("badge_thunder",
     MAP_ROCK_TUNNEL_1F,
     lambda b, vm: bool(b & BADGE_THUNDER)),

    # -----------------------------------------------------------------------
    # PHASE 4 — Thunder Badge → Rainbow Badge
    # -----------------------------------------------------------------------

    # 15 — entered Rock Tunnel (via Diglett's Cave detour or Route 9)
    ("rock_tunnel",
     MAP_LAVENDER_TOWN,
     lambda b, vm: MAP_ROCK_TUNNEL_1F in vm),

    # 16 — first visit to Lavender Town; need Silph Scope before clearing tower
    ("lavender_town_visit",
     MAP_CELADON_CITY,
     lambda b, vm: MAP_LAVENDER_TOWN in vm),

    # 17 — reached Celadon City; clear Game Corner Rocket Hideout for Silph Scope
    ("celadon_city",
     MAP_CELADON_GYM,
     lambda b, vm: MAP_CELADON_CITY in vm),

    # 18 — Rainbow Badge (Erika, Lv 29 Victreebel / Lv 24 Tangela / Lv 29 Vileplume)
    ("badge_rainbow",
     MAP_POKEMON_TOWER_1F,
     lambda b, vm: bool(b & BADGE_RAINBOW)),

    # -----------------------------------------------------------------------
    # PHASE 5 — Rainbow Badge → Soul Badge
    # -----------------------------------------------------------------------

    # 19 — returned to Pokemon Tower with Silph Scope; rescue Mr. Fuji for Poke Flute
    ("pokemon_tower",
     MAP_FUCHSIA_CITY,
     lambda b, vm: MAP_POKEMON_TOWER_1F in vm and bool(b & BADGE_RAINBOW)),

    # 20 — reached Fuchsia City; visit Safari Zone for HM Strength
    ("fuchsia_city",
     MAP_FUCHSIA_GYM,
     lambda b, vm: MAP_FUCHSIA_CITY in vm),

    # 21 — entered Safari Zone; find Gold Teeth → return to Warden for HM04 Strength
    ("safari_zone",
     MAP_FUCHSIA_GYM,
     lambda b, vm: MAP_SAFARI_ZONE in vm),

    # 22 — Soul Badge (Koga, Lv 37 Koffing / Lv 39 Muk / Lv 37 Koffing / Lv 43 Weezing)
    ("badge_soul",
     MAP_SAFFRON_CITY,
     lambda b, vm: bool(b & BADGE_SOUL)),

    # -----------------------------------------------------------------------
    # PHASE 6 — Soul Badge → Marsh Badge
    # -----------------------------------------------------------------------

    # 23 — reached Saffron City; infiltrate Silph Co. to defeat Giovanni
    ("saffron_city",
     MAP_SILPH_CO_1F,
     lambda b, vm: MAP_SAFFRON_CITY in vm),

    # 24 — entered Silph Co.; find Card Key (5F) and reach Giovanni (11F)
    ("silph_co",
     MAP_SAFFRON_GYM,
     lambda b, vm: MAP_SILPH_CO_1F in vm),

    # 25 — Marsh Badge (Sabrina, Lv 38 Kadabra / Lv 37 Mr. Mime / Lv 38 Venomoth / Lv 43 Alakazam)
    ("badge_marsh",
     MAP_SEAFOAM_1F,
     lambda b, vm: bool(b & BADGE_MARSH)),

    # -----------------------------------------------------------------------
    # PHASE 7 — Marsh Badge → Volcano Badge
    # -----------------------------------------------------------------------

    # 26 — entered Seafoam Islands (Surf + Strength required); passage to Cinnabar
    ("seafoam_islands",
     MAP_CINNABAR_ISLAND,
     lambda b, vm: MAP_SEAFOAM_1F in vm),

    # 27 — reached Cinnabar Island; enter Pokemon Mansion for the Secret Key
    ("cinnabar_island",
     MAP_POKEMON_MANSION,
     lambda b, vm: MAP_CINNABAR_ISLAND in vm),

    # 28 — entered Pokemon Mansion; find Secret Key to unlock Cinnabar Gym
    ("pokemon_mansion",
     MAP_CINNABAR_GYM,
     lambda b, vm: MAP_POKEMON_MANSION in vm),

    # 29 — Volcano Badge (Blaine, Lv 42 Growlithe / Lv 40 Ponyta / Lv 42 Rapidash / Lv 47 Arcanine)
    ("badge_volcano",
     MAP_VIRIDIAN_GYM,
     lambda b, vm: bool(b & BADGE_VOLCANO)),

    # -----------------------------------------------------------------------
    # PHASE 8 — Volcano Badge → Earth Badge → Champion
    # -----------------------------------------------------------------------

    # 30 — Earth Badge (Giovanni, Lv 45 Rhyhorn / Lv 42 Dugtrio / Lv 44 Nidoqueen /
    #                   Lv 45 Nidoking / Lv 50 Rhydon)
    ("badge_earth",
     MAP_VICTORY_ROAD_1F,
     lambda b, vm: bool(b & BADGE_EARTH)),

    # 31 — entered Victory Road; solve boulder puzzles with Strength
    ("victory_road",
     MAP_E4_LORELEI,
     lambda b, vm: MAP_VICTORY_ROAD_1F in vm),

    # 32 — entered Elite Four (Lorelei's chamber)
    ("elite_four",
     MAP_HALL_OF_FAME,
     lambda b, vm: MAP_E4_LORELEI in vm),

    # 33 — Hall of Fame!  Game complete.
    ("hall_of_fame",
     MAP_HALL_OF_FAME,
     lambda b, vm: MAP_HALL_OF_FAME in vm),
]

NUM_MILESTONES = len(MILESTONES)  # 34

# Reward given each time a new milestone is unlocked.
MILESTONE_BONUS = 10.0

# ---------------------------------------------------------------------------
# Hint strings (parallel to MILESTONES)
#
# Each string describes what the agent should do to reach the NEXT milestone.
# Used by stream.py / watch.py for the goal display and by the RL agent's
# info dict so human observers can follow along.
# ---------------------------------------------------------------------------

MILESTONE_HINTS = [
    # 0 start
    "Head north out of Pallet Town onto Route 1. Professor Oak will stop you "
    "and take you to his lab — pick your starter Pokemon and beat the rival.",

    # 1 route_1
    "Walk north to Viridian City. Visit the Poke Mart — the clerk gives you "
    "Oak's Parcel. Backtrack to Pallet Town and deliver it to Professor Oak "
    "to receive the Pokédex. Return north; the old man now lets you pass.",

    # 2 viridian_city
    "Enter Viridian Forest and navigate the maze north. Battle the Bug "
    "Catchers for easy XP. Exit north to reach Pewter City.",

    # 3 viridian_forest
    "You're in Pewter City. Challenge Brock at the Pewter Gym. "
    "He uses Rock/Ground types (Geodude Lv12, Onix Lv14). "
    "Water and Grass moves are super effective.",

    # 4 pewter_city
    "Brock is next! Use Water or Grass attacks — Butterfree's Confusion "
    "also works. After winning, head east on Route 3 toward Mt. Moon.",

    # 5 badge_boulder
    "Boulder Badge earned! Travel east on Route 3 fighting trainers. "
    "Enter Mt. Moon and navigate to B2F. Pick either the Dome Fossil "
    "(Kabuto) or Helix Fossil (Omanyte) from the Super Nerd.",

    # 6 mt_moon
    "Navigate deeper into Mt. Moon. Head down to B1F then B2F. "
    "Fight the Super Nerd on B2F to claim a fossil, then exit east "
    "onto Route 4 toward Cerulean City.",

    # 7 mt_moon_b2f
    "Almost through Mt. Moon! Exit east onto Route 4 and continue to "
    "Cerulean City. Heal at the Pokemon Center, then challenge Misty.",

    # 8 cerulean_city
    "Challenge Misty at Cerulean Gym (Staryu Lv18, Starmie Lv21). "
    "Electric and Grass moves are super effective. After winning, "
    "head north on Route 24 across Nugget Bridge toward Bill.",

    # 9 badge_cascade
    "Cascade Badge earned! Head north on Route 24, defeat all 5 trainers "
    "on Nugget Bridge, then continue east on Route 25 to Bill's Sea Cottage.",

    # 10 nugget_bridge
    "Find Bill's Sea Cottage on Route 25. Talk to him twice to reverse his "
    "DNA experiment — he rewards you with the S.S. Ticket. Head south "
    "through Routes 25 and 5/6 via the Underground Path to Vermilion City.",

    # 11 bills_cottage
    "Head to Vermilion City. Visit the Pokemon Fan Club for a Bike Voucher "
    "and the northwest house for the Old Rod. Then board the S.S. Anne "
    "at the harbor using the S.S. Ticket.",

    # 12 vermilion_city
    "Explore the S.S. Anne — battle your rival on 2F. Find the Captain "
    "in his cabin, help him with his seasickness, and receive HM01 Cut. "
    "Disembark, teach Cut to a Pokemon, and cut the tree blocking the gym.",

    # 13 ss_anne
    "Challenge Lt. Surge at Vermilion Gym (Voltorb Lv21, Pikachu Lv18, "
    "Raichu Lv24). Solve the trash can puzzle to open the gym doors. "
    "Ground-type moves are immune to Electric — Dig is very effective.",

    # 14 badge_thunder
    "Thunder Badge earned! Go east through Diglett's Cave to Route 2, "
    "then south through Rock Tunnel on Route 10. Use HM Flash to "
    "reduce accuracy of wild Pokemon in the dark tunnel.",

    # 15 rock_tunnel
    "Exit Rock Tunnel south onto Route 10 and head north to Lavender Town. "
    "Visit the Pokemon Tower — but you can't clear it yet without the "
    "Silph Scope. Head west on Route 8 toward Celadon City.",

    # 16 lavender_town_visit
    "You're in Celadon City! Find the Coin Case in the Restaurant, then "
    "locate the hidden switch in the Game Corner to access Team Rocket "
    "Hideout. Fight your way to Giovanni on B4F to get the Silph Scope.",

    # 17 celadon_city
    "Defeat Erika at Celadon Gym (Victreebel Lv29, Tangela Lv24, "
    "Vileplume Lv29) — Fire, Ice, and Flying moves are effective. "
    "Then return to Lavender Town with the Silph Scope to clear "
    "Pokemon Tower and rescue Mr. Fuji.",

    # 18 badge_rainbow
    "Rainbow Badge earned! Return to Lavender Town. Use the Silph Scope "
    "to reveal and battle the Ghost Pokemon in Pokemon Tower. "
    "Reach the 7F to defeat the Rockets and receive the Poke Flute "
    "from Mr. Fuji.",

    # 19 pokemon_tower
    "Pokemon Tower cleared! Use the Poke Flute to wake the Snorlax "
    "on Route 12 or Route 16. Head south to Fuchsia City — you'll need "
    "HM Surf (get it from the Safari Zone Warden) to continue later.",

    # 20 fuchsia_city
    "Enter the Safari Zone and find the Warden's Gold Teeth. Return them "
    "to the Safari Zone Warden in Fuchsia City to receive HM04 Strength. "
    "Then challenge Koga at Fuchsia Gym.",

    # 21 safari_zone
    "Challenge Koga at Fuchsia Gym (Koffing Lv37, Muk Lv39, Koffing Lv37, "
    "Weezing Lv43). The gym has invisible walls — navigate carefully. "
    "Psychic and Ground moves are effective against Poison types.",

    # 22 badge_soul
    "Soul Badge earned! Head north then east to Saffron City. Give a "
    "guard a drink (Lemonade, Soda Pop, or Fresh Water from Celadon "
    "Department Store rooftop) to unlock the city gates.",

    # 23 saffron_city
    "Infiltrate Silph Co. HQ — Team Rocket has taken over. Find the "
    "Card Key on 5F to open locked doors. Battle your rival on 7F "
    "and defeat Giovanni on 11F. The president gives you the Master Ball!",

    # 24 silph_co
    "Challenge Sabrina at Saffron Gym (Kadabra Lv38, Mr. Mime Lv37, "
    "Venomoth Lv38, Alakazam Lv43). Ghost and Bug moves resist Psychic — "
    "use Dark-era workarounds or raw power. After winning, surf southwest "
    "via Routes 19-20 to reach the Seafoam Islands.",

    # 25 badge_marsh
    "Marsh Badge earned! Fly to Fuchsia City and surf south on Routes 19-20. "
    "Navigate through the Seafoam Islands using Surf and Strength to push "
    "boulders. Articuno is catchable on B4F. Exit west to Cinnabar Island.",

    # 26 seafoam_islands
    "You've reached Cinnabar Island! Visit Cinnabar Lab to revive your "
    "fossil. Enter the Pokemon Mansion and navigate to B1F to find the "
    "Secret Key needed to unlock Cinnabar Gym.",

    # 27 cinnabar_island
    "Navigate the Pokemon Mansion (watch out for strong wild Pokemon). "
    "Find the Secret Key on B1F. Return to Cinnabar Gym and use the "
    "key to unlock the doors. Challenge Blaine.",

    # 28 pokemon_mansion
    "Challenge Blaine at Cinnabar Gym (Growlithe Lv42, Ponyta Lv40, "
    "Rapidash Lv42, Arcanine Lv47). Quiz doors can be bypassed by "
    "answering incorrectly and fighting the trainer instead. "
    "Water and Rock moves are very effective.",

    # 29 badge_volcano
    "Volcano Badge earned! Fly back to Viridian City. The Viridian Gym "
    "is finally open — challenge Giovanni (Rhyhorn Lv45, Dugtrio Lv42, "
    "Nidoqueen Lv44, Nidoking Lv45, Rhydon Lv50). "
    "Water and Grass destroy his Ground team.",

    # 30 badge_earth
    "All 8 Badges! Head west to Route 22 — battle your rival there. "
    "Then go north through Route 23, showing all 8 badges to the guards. "
    "Enter Victory Road and solve the boulder puzzles with Strength.",

    # 31 victory_road
    "Navigate Victory Road (3 floors of boulder puzzles and tough trainers). "
    "Moltres is on 2F if you want to catch it. Stock up on Full Restores "
    "and Revives at the Indigo Plateau Poke Mart before the Elite Four.",

    # 32 elite_four
    "The Elite Four! Face them in order: Lorelei (Ice, use Electric/Rock), "
    "Bruno (Fighting, use Psychic/Flying), Agatha (Ghost, use Psychic), "
    "Lance (Dragon, use Ice). Then defeat Champion Blue. No breaks allowed!",

    # 33 hall_of_fame
    "You're the Champion! Post-game: Cerulean Cave (northwest of Cerulean "
    "City via Surf) is now open. Mewtwo waits at Lv70 on B1F. "
    "Use the Master Ball for a guaranteed catch.",
]

assert len(MILESTONE_HINTS) == len(MILESTONES), (
    f"MILESTONE_HINTS length {len(MILESTONE_HINTS)} != "
    f"MILESTONES length {len(MILESTONES)}"
)


# ---------------------------------------------------------------------------
# GameGuide class
# ---------------------------------------------------------------------------

class GameGuide:
    """Deterministic rule-based guide agent.

    Usage inside PokemonBlueEnv:

        # __init__
        self.guide = GameGuide()

        # _compute_reward  (after updating visited_maps)
        guide_bonus = self.guide.update(badges, self.visited_maps)
        reward += guide_bonus

        # _get_memory_features
        features[15] = self.guide.progress
        features[16] = self.guide.target_map_id / 255.0
    """

    def __init__(self):
        # milestone_index persists across episodes — once a milestone is
        # unlocked it stays unlocked, just like visited_maps.
        self.milestone_index: int = 0
        self._last_rewarded: int = -1   # highest milestone already rewarded

    def reset(self):
        """Episode reset hook — intentionally does NOT reset milestone state."""
        pass   # state intentionally preserved

    def update(self, badges: int, visited_maps: set) -> float:
        """Advance the guide and return a milestone bonus if progress was made.

        Args:
            badges:       raw bitmask from memory address 0xD356.
            visited_maps: global set of map IDs ever visited.

        Returns:
            Bonus reward (0.0 normally; MILESTONE_BONUS per new milestone).
        """
        while self.milestone_index < NUM_MILESTONES - 1:
            _name, _target, detector = MILESTONES[self.milestone_index + 1]
            if detector(badges, visited_maps):
                self.milestone_index += 1
            else:
                break

        bonus = 0.0
        if self.milestone_index > self._last_rewarded:
            bonus = MILESTONE_BONUS * (self.milestone_index - self._last_rewarded)
            self._last_rewarded = self.milestone_index

        return bonus

    # ------------------------------------------------------------------
    # Observation features (read-only properties)
    # ------------------------------------------------------------------

    @property
    def progress(self) -> float:
        """Normalised game progress in [0, 1].  Slot [15] of memory_features."""
        return self.milestone_index / (NUM_MILESTONES - 1)

    @property
    def target_map_id(self) -> int:
        """Map ID the agent should head toward next.  Slot [16]."""
        _name, target_map, _det = MILESTONES[self.milestone_index]
        return target_map

    @property
    def milestone_name(self) -> str:
        """Human-readable milestone label (logging / debug)."""
        name, _t, _d = MILESTONES[self.milestone_index]
        return name

    @property
    def milestone_hint(self) -> str:
        """Full walkthrough hint for the current stage (human display)."""
        return MILESTONE_HINTS[self.milestone_index]
