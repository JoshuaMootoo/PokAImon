# Pokemon Blue — Game Boy WRAM memory addresses and read helpers.
# All addresses are in the System Bus / WRAM space of the GB ROM.

# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------

MAP_ID   = 0xD35E  # current map identifier
PLAYER_X = 0xD362  # tile X coordinate
PLAYER_Y = 0xD361  # tile Y coordinate

BADGES = 0xD356    # bitmask: bit 0=Boulder, 1=Cascade, 2=Thunder, 3=Rainbow,
                   #          4=Soul, 5=Marsh, 6=Volcano, 7=Earth

MONEY_0 = 0xD347   # most-significant BCD byte  (e.g. $123456 -> 0x12, 0x34, 0x56)
MONEY_1 = 0xD348
MONEY_2 = 0xD349

PARTY_COUNT = 0xD163  # number of Pokemon in party (0–6)
BATTLE_FLAG = 0xD057  # 0=none, 1=wild battle, 2=trainer battle

POKEDEX_OWNED_START = 0xD2F7  # 19 bytes, bit-packed (Pokemon #1 = bit 0 of byte 0)
POKEDEX_SEEN_START  = 0xD30A  # 19 bytes, same layout

# Base addresses for each of the 6 party slots (44 bytes each)
PARTY_ADDRS = [0xD16B, 0xD197, 0xD1C3, 0xD1EF, 0xD21B, 0xD247]

# ---------------------------------------------------------------------------
# Per-party-slot byte offsets
# ---------------------------------------------------------------------------
OFF_SPECIES   = 0x00
OFF_CUR_HP_HI = 0x01   # current HP, big-endian 2 bytes
OFF_CUR_HP_LO = 0x02
OFF_STATUS    = 0x04   # bit flags: sleep(0-2), poison(3), burn(4), freeze(5), paralysis(6)
# 0x07 = catch rate (1B), 0x08-0x0B = move IDs (4×1B), 0x0C-0x0D = OT ID (2B)
OFF_XP_0      = 0x0E   # XP, big-endian 3 bytes (was wrongly 0x07 = catch rate)
OFF_XP_1      = 0x0F
OFF_XP_2      = 0x10
OFF_LEVEL     = 0x21   # actual current level (NOT +3, which is the status byte)
OFF_MAX_HP_HI = 0x22   # max HP, big-endian 2 bytes
OFF_MAX_HP_LO = 0x23

# ---------------------------------------------------------------------------
# Helper functions
# All accept a PyBoy memory object (pyboy.memory) which supports [addr] reads.
# ---------------------------------------------------------------------------

def read_bcd(memory, start_addr: int, num_bytes: int) -> int:
    """Decode a BCD-encoded integer (used for money).
    Each byte stores two decimal digits (high nibble = tens, low = ones).
    """
    value = 0
    for i in range(num_bytes):
        byte = memory[start_addr + i]
        value = value * 100 + (byte >> 4) * 10 + (byte & 0x0F)
    return value


def read_bit(memory, addr: int, bit: int) -> bool:
    """Return True if the given bit (0 = LSB) is set at addr."""
    return bool(memory[addr] & (1 << bit))


def count_bits(memory, start_addr: int, num_bytes: int) -> int:
    """Count the number of set bits across a contiguous byte range."""
    total = 0
    for i in range(num_bytes):
        total += bin(memory[start_addr + i]).count('1')
    return total


def read_party_pokemon(memory, slot_index: int) -> dict:
    """Return a dict with key stats for one party slot (0-indexed).

    Returns zeros for empty/invalid slots.
    """
    if slot_index < 0 or slot_index > 5:
        return {"species": 0, "cur_hp": 0, "max_hp": 0, "level": 0, "status": 0, "xp": 0}

    base = PARTY_ADDRS[slot_index]
    species = memory[base + OFF_SPECIES]

    cur_hp = (memory[base + OFF_CUR_HP_HI] << 8) | memory[base + OFF_CUR_HP_LO]
    max_hp = (memory[base + OFF_MAX_HP_HI] << 8) | memory[base + OFF_MAX_HP_LO]
    level  = memory[base + OFF_LEVEL]
    status = memory[base + OFF_STATUS]
    xp     = ((memory[base + OFF_XP_0] << 16) |
               (memory[base + OFF_XP_1] << 8)  |
                memory[base + OFF_XP_2])

    return {
        "species": species,
        "cur_hp":  cur_hp,
        "max_hp":  max_hp,
        "level":   level,
        "status":  status,
        "xp":      xp,
    }


def read_badges(memory) -> int:
    """Return the number of badges obtained (0–8)."""
    return bin(memory[BADGES]).count('1')


def is_in_battle(memory) -> bool:
    """Return True if the player is currently in a battle."""
    return memory[BATTLE_FLAG] != 0


def read_party_hp_fraction(memory) -> float:
    """Return the mean HP fraction across all alive party members.

    Returns 1.0 if no party Pokemon are loaded (avoids division by zero).
    """
    party_size = memory[PARTY_COUNT]
    if party_size == 0:
        return 1.0

    total_cur = 0
    total_max = 0
    for i in range(min(party_size, 6)):
        mon = read_party_pokemon(memory, i)
        if mon["max_hp"] > 0:
            total_cur += mon["cur_hp"]
            total_max += mon["max_hp"]

    if total_max == 0:
        return 1.0
    return total_cur / total_max


def read_party_level_sum(memory) -> int:
    """Return the sum of all party Pokemon levels."""
    party_size = memory[PARTY_COUNT]
    total = 0
    for i in range(min(party_size, 6)):
        total += read_party_pokemon(memory, i)["level"]
    return total


def read_party_xp_total(memory) -> int:
    """Return the total XP across all party Pokemon.

    XP increases with every battle win (not just on level-up), giving the
    agent much more frequent reward signal for actually defeating enemies.
    """
    party_size = memory[PARTY_COUNT]
    total = 0
    for i in range(min(party_size, 6)):
        total += read_party_pokemon(memory, i)["xp"]
    return total
