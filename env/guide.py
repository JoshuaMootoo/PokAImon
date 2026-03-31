"""Pokemon Blue Game Guide — rule-based progression tracker.

This is the second "AI" in the two-agent system.  It is entirely
deterministic and requires no training — it simply reads the current
game state, works out which milestone has been reached, and tells the
PPO learner:

  1. How far through the game it is (normalised progress scalar)
  2. Which map it should be heading toward next (target map hint)
  3. A one-time bonus reward every time a new milestone is unlocked

Milestone state intentionally persists across episodes (like
tile_visit_count / visited_maps) so the guide always points toward
genuinely new territory rather than resetting its knowledge each time.

Walkthrough reference: bulbapedia.bulbagarden.net/wiki/
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
MAP_VIRIDIAN_FOREST  = 37
MAP_MT_MOON_1F       = 38
MAP_MT_MOON_B1F      = 39
MAP_MT_MOON_B2F      = 40
MAP_ROCK_TUNNEL_1F   = 41
MAP_POKEMON_TOWER_1F = 43
MAP_SEAFOAM_1F       = 51
MAP_VICTORY_ROAD_1F  = 57
MAP_SS_ANNE          = 64   # exterior / corridor maps start here
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
# Each entry is a tuple:
#   (name,  next_target_map_id,  detector_fn)
#
# detector_fn(badge_bitmask: int, visited_maps: set[int]) -> bool
#   Returns True once this milestone has been reached.
#
# Milestones are ordered — the guide advances through them linearly.
# The detector for milestone N is evaluated only after milestone N-1 is done.
# ---------------------------------------------------------------------------

MILESTONES = [
    # 0 — game begins in Pallet Town; head north to Viridian City
    ("start",
     MAP_VIRIDIAN_CITY,
     lambda b, vm: True),

    # 1 — reached Viridian City; visit Pokemart to receive Oak's Parcel,
    #     then return it to Oak.  The old man blocking Route 2 north will
    #     move once the parcel is delivered.
    ("viridian_city",
     MAP_VIRIDIAN_FOREST,
     lambda b, vm: MAP_VIRIDIAN_CITY in vm),

    # 2 — entered Viridian Forest; old man has moved, parcel delivered.
    #     Navigate the forest maze north to Pewter City.
    ("viridian_forest",
     MAP_PEWTER_CITY,
     lambda b, vm: MAP_VIRIDIAN_FOREST in vm),

    # 3 — reached Pewter City; challenge Brock in the Pewter Gym.
    ("pewter_city",
     MAP_PEWTER_GYM,
     lambda b, vm: MAP_PEWTER_CITY in vm),

    # 4 — Boulder Badge earned.  Head east on Route 3 toward Mt. Moon.
    ("badge_boulder",
     MAP_MT_MOON_1F,
     lambda b, vm: bool(b & BADGE_BOULDER)),

    # 5 — entered Mt. Moon.  Navigate through to Route 4 east exit.
    ("mt_moon",
     MAP_CERULEAN_CITY,
     lambda b, vm: MAP_MT_MOON_1F in vm),

    # 6 — reached Cerulean City.  Challenge Misty; then head north to Bill
    #     on Route 25 to get the S.S. Ticket before going to Vermilion.
    ("cerulean_city",
     MAP_CERULEAN_GYM,
     lambda b, vm: MAP_CERULEAN_CITY in vm),

    # 7 — Cascade Badge earned.  Head south/east to Vermilion City.
    ("badge_cascade",
     MAP_VERMILION_CITY,
     lambda b, vm: bool(b & BADGE_CASCADE)),

    # 8 — reached Vermilion City.  Board S.S. Anne to get HM01 Cut,
    #     then challenge Lt. Surge.
    ("vermilion_city",
     MAP_VERMILION_GYM,
     lambda b, vm: MAP_VERMILION_CITY in vm),

    # 9 — Thunder Badge earned.  Head east via Diglett's Cave / Rock Tunnel
    #     toward Lavender Town, then west to Celadon City.
    ("badge_thunder",
     MAP_CELADON_CITY,
     lambda b, vm: bool(b & BADGE_THUNDER)),

    # 10 — reached Celadon City.  Defeat Erika; clear Rocket Hideout for
    #      the Silph Scope.
    ("celadon_city",
     MAP_CELADON_GYM,
     lambda b, vm: MAP_CELADON_CITY in vm),

    # 11 — Rainbow Badge earned.  Head to Lavender Town with the Silph
    #      Scope to clear Pokemon Tower and rescue Mr. Fuji (Poke Flute).
    ("badge_rainbow",
     MAP_LAVENDER_TOWN,
     lambda b, vm: bool(b & BADGE_RAINBOW)),

    # 12 — reached Lavender Town.  Climb Pokemon Tower to the 7F, defeat
    #      the Rockets, and get the Poke Flute from Mr. Fuji.
    ("lavender_town",
     MAP_FUCHSIA_CITY,
     lambda b, vm: MAP_LAVENDER_TOWN in vm),

    # 13 — reached Fuchsia City.  Get HM04 Strength from the Safari Zone
    #      Warden; challenge Koga in the Fuchsia Gym.
    ("fuchsia_city",
     MAP_FUCHSIA_GYM,
     lambda b, vm: MAP_FUCHSIA_CITY in vm),

    # 14 — Soul Badge earned.  Head to Saffron City (give a beverage to
    #      the gate guard if not done already); clear Silph Co.
    ("badge_soul",
     MAP_SAFFRON_CITY,
     lambda b, vm: bool(b & BADGE_SOUL)),

    # 15 — reached Saffron City.  Clear Silph Co. for the Master Ball;
    #      then challenge Sabrina.
    ("saffron_city",
     MAP_SAFFRON_GYM,
     lambda b, vm: MAP_SAFFRON_CITY in vm),

    # 16 — Marsh Badge earned.  Surf west along Routes 19-20 to reach
    #      Cinnabar Island.  Explore Pokemon Mansion for the Secret Key.
    ("badge_marsh",
     MAP_CINNABAR_ISLAND,
     lambda b, vm: bool(b & BADGE_MARSH)),

    # 17 — reached Cinnabar Island.  Get Secret Key from Pokemon Mansion;
    #      challenge Blaine.
    ("cinnabar_island",
     MAP_CINNABAR_GYM,
     lambda b, vm: MAP_CINNABAR_ISLAND in vm),

    # 18 — Volcano Badge earned.  Return north to Viridian City and
    #      challenge Giovanni to earn the final badge.
    ("badge_volcano",
     MAP_VIRIDIAN_GYM,
     lambda b, vm: bool(b & BADGE_VOLCANO)),

    # 19 — Earth Badge earned!  All 8 badges collected.  Head to Route 23
    #      and through Victory Road to the Indigo Plateau.
    ("badge_earth",
     MAP_VICTORY_ROAD_1F,
     lambda b, vm: bool(b & BADGE_EARTH)),

    # 20 — entered Victory Road.  Solve the boulder puzzles to exit north.
    ("victory_road",
     MAP_E4_LORELEI,
     lambda b, vm: MAP_VICTORY_ROAD_1F in vm),

    # 21 — entered Elite Four.  Defeat Lorelei, Bruno, Agatha, Lance, then
    #      the Champion to reach the Hall of Fame.
    ("elite_four",
     MAP_HALL_OF_FAME,
     lambda b, vm: MAP_E4_LORELEI in vm),

    # 22 — Hall of Fame!  Game completed.
    ("hall_of_fame",
     MAP_HALL_OF_FAME,
     lambda b, vm: MAP_HALL_OF_FAME in vm),
]

NUM_MILESTONES = len(MILESTONES)  # 23

# Reward given each time a new milestone is unlocked.
MILESTONE_BONUS = 10.0


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

        # _get_memory_features  (replace the reserved [15] slot)
        features[15] = self.guide.progress
        features[16] = self.guide.target_map_id / 255.0
    """

    def __init__(self):
        # milestone_index persists across episodes — once a milestone is
        # unlocked it stays unlocked, just like visited_maps.
        self.milestone_index: int = 0
        self._last_rewarded: int = -1   # highest milestone already rewarded

    def reset(self):
        """Episode reset hook — intentionally does NOT reset milestone state.

        Call this from PokemonBlueEnv.reset() so the interface is consistent,
        but the guide's progress memory is preserved across all episodes.
        """
        pass   # state intentionally preserved

    def update(self, badges: int, visited_maps: set) -> float:
        """Advance the guide and return a milestone bonus if progress was made.

        Call once per step, after visited_maps has been updated.

        Args:
            badges:       raw bitmask from memory address 0xD356.
            visited_maps: global set of map IDs ever visited (env instance).

        Returns:
            Bonus reward (0.0 normally; MILESTONE_BONUS per new milestone).
        """
        # Walk forward through milestones, advancing as far as the game state
        # allows.  We stop at the first milestone whose detector returns False.
        while self.milestone_index < NUM_MILESTONES - 1:
            _name, _target, detector = MILESTONES[self.milestone_index + 1]
            if detector(badges, visited_maps):
                self.milestone_index += 1
            else:
                break

        # Fire a one-time bonus for every newly reached milestone.
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
        """Map ID the agent should currently be heading toward.  Slot [16]."""
        _name, target_map, _det = MILESTONES[self.milestone_index]
        return target_map

    @property
    def milestone_name(self) -> str:
        """Human-readable label for the current milestone (logging / debug)."""
        name, _t, _d = MILESTONES[self.milestone_index]
        return name
