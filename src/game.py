from __future__ import annotations

import random
from enum import Enum

import numpy as np

random.seed(5454)


class Player(Enum):
    NONE = 0
    PLAYER_1 = 1
    PLAYER_2 = 2


# class GameAction:
#   def __init__(self, rejected, ):
#     self.

class GameAction:
    def __init__(self, discard, placement1, placement2):
        self.discard = discard
        self.placements = [placement1, placement2]


class GameState:


    def __init__(self):

        self.drawer = Drawer()
        self.board: HexBoard = HexBoard()
        self.cur_player = Player.PLAYER_1
        self.turn = 0
        self.hand = self._get_random_hand_from_template()
        self.game_over = False
        self.decks = [None] * 3
        self.hands = [None] * 3
        self.action_tokens = {"battle"}


    def get_edges(self):
        edges = []
        for i, tile in enumerate(self.board.tiles):
            for n in tile.neighbors:
                if n != -1:
                    edges.append([i, n])
        return edges




    def _get_base_deck(self):
        deck = []
        for i in range(4):
            deck.append("warrior")
        for i in range(4):
            deck.append("assasin")
        for i in range(3):
            deck.append("brawler")
        for i in range(2):
            deck.append("gladiator")
        for i in range(3):
            deck.append("anti-swarm")
        for i in range(2):
            deck.append("champion")
        for i in range(3):
            deck.append("though_one")
        for i in range(5):
            deck.append("battle")

        random.shuffle(deck)
        deck.append("HQ")
        return deck






    def _get_random_hand_from_template(self):
        keys = list(TileInfo.unit_template_library.keys())

        return (
            random.choice(keys),
            random.choice(keys),
            random.choice(keys)
        )

    def get_data(self, sample_size):
        X = np.zeros((sample_size, 19, TileInfo.data_len), dtype=np.float32)
        Y = np.zeros((sample_size, 1), dtype=np.float32)

        for i in range(sample_size):
            self.reset()
            self.randomly_populate()
            for j, tile in enumerate(self.board.tiles):
                TileInfo.load_to_np(tile.tileinfo, X[i, j, :])

            Y[i] = self.battle()

        return X, Y
    def get_data2(self, sample_size):
        X = np.zeros((sample_size, 19, TileInfo.data_len), dtype=np.float32)
        Y = np.zeros((sample_size, 1), dtype=np.float32)
        d = [None] * sample_size
        for i in range(sample_size):
            self.reset()
            self.randomly_populate()


            for j, tile in enumerate(self.board.tiles):
                TileInfo.load_to_np(tile.tileinfo, X[i, j, :])

            Y[i] = self.battle()
            drawed = []
            drawed.append(drawer.get_board(self.board, config={"hide_dead": False}))
            drawed.append(drawer.get_board(self.board))
            d[i] = drawed

        return X, Y, d



    def add_to_hand(self, hand, token):
        for i in range(3):
            if hand[i] == None:
                hand[i] = token
                return True
        return False
    def reset(self):
        for item in self.board.tiles:
            item.tileinfo.alive = False
            item.tileinfo.player = Player.NONE
        self.turn = 0
        self.cur_player = random.choice([Player.PLAYER_1, Player.PLAYER_2])
        self.decks[1] = self._get_base_deck()
        self.decks[2] = self._get_base_deck()
        self.hands[1] = [None, None, None]
        self.hands[2] = [None, None, None]


        self.add_to_hand(self.hands[self.cur_player.value], self.decks[self.cur_player.value].pop())



        return self.cur_player, self.hands[self.cur_player.value], False

    def step(self, action: dict, config: dict):

        draw = False
        if config:
            draw = config.get("draw", False)
        drawed = []
        # Pobieramy rękę aktualnego gracza
        hand = self.hands[self.cur_player.value]

        # 1. Obsługa odrzucenia (Discard)
        discard_idx = action.get("discard_idx", -1)
        # Weryfikujemy, czy indeks jest poprawny i czy element istnieje na ręce
        if 0 <= discard_idx < len(hand):
            # Zapamiętujemy token do usunięcia, ale usuwamy go PO placementach,
            # aby nie zepsuć indeksów (hand_idx) dla placement1 i placement2!
            hand[discard_idx] = None


        # 2. Zbieramy placementy ze słownika do bezpiecznej listy do przeiterowania
        placements_to_process = []
        for p_name in ["placement1", "placement2"]:
            p = action.get(p_name)
            # Sprawdzamy czy placement istnieje i czy nie ma wartości domyślnych/pustych (-1)
            if p and p.get("hand_idx", -1) != -1:
                placements_to_process.append(p)

        # 3. Przetwarzamy placementy
        for p in placements_to_process:
            hand_idx = p["hand_idx"]
            board_idx = p["board_idx"]
            rot = p["rot"]

            # Pobieramy klucz żetonu z ręki na podstawie przekazanego indeksu
            key = hand[hand_idx]

            # Sprawdzamy czy to żeton natychmiastowy (Instant Token)
            if key in self.action_tokens:
                if key == "battle":
                    self.battle()

                if draw:
                    drawed.append(self.drawer.get_board(self.board, config = {"hide_dead": False}))
                    for tile in self.board.tiles:
                        if not tile.tileinfo.alive:
                            tile.tileinfo.player = Player.NONE
                # Tutaj możesz dodać obsługę snajpera, ruchu, granatu itp.

                # Żetony natychmiastowe znikają z ręki po zagraniu
                hand[hand_idx] = None
            else:
                # Wystawienie jednostki/modułu na planszę
                tile = self.board.tiles[board_idx]
                tile.tileinfo = TileInfo.from_template(key, self.cur_player, rot)

                # Żeton ląduje na planszy, więc znika z ręki
                hand[hand_idx] = None

        # # 4. Teraz bezpiecznie czyścimy rękę z odrzuconego żetonu oraz tych zagranych
        # if token_to_discard and token_to_discard in hand:
        #     hand.remove(token_to_discard)

        # Usuwamy z ręki "dziury" (None) powstałe po zagraniu żetonów planszowych/natychmiastowych
        #self.hands[self.cur_player.value] = [token for token in hand if token is not None]

        # 5. Logika dociągu kart (bez zmian)
        max_draw = 1
        if self.turn == 2:
            max_draw = 2
        elif self.turn > 2:
            max_draw = 3

        print(f"{max_draw=} {self.turn=}")
        # 6. Zmiana gracza
        self.cur_player = Player.PLAYER_1 if self.cur_player == Player.PLAYER_2 else Player.PLAYER_2

        # Pobieramy rękę i indeks NOWEGO aktualnego gracza
        next_hand = self.hands[self.cur_player.value]
        idx = self.cur_player.value
        i = 0

        # Dociąganie żetonów z talii nowego gracza
        added = True
        deck: list = self.decks[self.cur_player.value]
        while i < max_draw and added:
            i += 1

            added = self.add_to_hand(next_hand, deck[-1])

            if added:
                deck.pop()
            else:
                break

        # Inkrementacja tury gry (warto to robić globalnie w silniku)
        self.turn += 1

        if draw:
            drawed.append(self.drawer.get_board(self.board))


        return self.cur_player, next_hand, False, drawed





    def randomly_populate(self):
        # 1. losowy pierwszy gracz
        cur_player = random.choice([Player.PLAYER_1, Player.PLAYER_2])

        free_indexes = list()
        for i, item in enumerate(self.board.tiles):
            if item.tileinfo.player == Player.NONE or not item.tileinfo.alive:
                free_indexes.append(i)

        keys = list(TileInfo.unit_template_library.keys())
        keys.remove("none")

        while free_indexes:
            idx = random.choice(free_indexes)
            free_indexes.remove(idx)
            key = random.choice(keys)

            TileInfo.insert_TileInfo(self.board.tiles[idx], random.randint(0, 5), cur_player, key)
            cur_player = Player.PLAYER_2 if cur_player == Player.PLAYER_1 else Player.PLAYER_1

    def battle(self):
        max_initiative = 0
        for tile in self.board.tiles:
            initiative = 0

            for i in range(5, 0, -1):
                if tile.tileinfo.initiatives[i] > 0:
                    initiative = i
                    break

            max_initiative = max(max_initiative, initiative)
        debug = False
        log = []
        for i in range(max_initiative, -1, -1):
            for tile in self.board.tiles:
                tile_info: TileInfo = tile.tileinfo
                if not tile_info.alive or tile_info.player == Player.NONE or not tile_info.initiatives[i]:
                    continue

                playerTarget = Player.PLAYER_2 if tile_info.player == Player.PLAYER_1 else Player.PLAYER_1
                for j, attack in enumerate(tile_info.attacks):
                    if attack > 0:
                        idx = tile.neighbors[j]
                        if idx == -1:
                            continue
                        target_tile: Tile = self.board.tiles[idx]
                        if target_tile.tileinfo.player == playerTarget and target_tile.tileinfo.alive:
                            target_tile.tileinfo.damage_taken += attack
                            if debug:
                                log.append(
                                    f"[INIT {i}] "
                                    f"P{tile_info.player.value} ({tile.id}) "
                                    f"-> P{playerTarget.value} ({idx}) | "
                                    f"DIR {j} | "
                                    f"DMG {attack} | "
                                    f"HP {target_tile.tileinfo.hp} |  "
                                    f"DMG_TAKEN {target_tile.tileinfo.damage_taken}"
                                )

            for tile in self.board.tiles:
                t: TileInfo = tile.tileinfo
                if not t.alive or t.player == Player.NONE:
                    continue

                before = t.alive
                t.ensure_token()
                after = t.alive

                if debug and before != after:
                    log.append(
                        f"[DEATH] P{t.player.value} @ tile {tile.id} "
                        f"(HP | DMG_TAKEN: {t.hp} | {t.damage_taken})"
                    )
        if debug:
            print("\n".join(log))

        player1_count = 0
        player2_count = 0

        for tile in self.board.tiles:
            if tile.tileinfo.alive and tile.tileinfo.player.value > 0:
                if tile.tileinfo.player == Player.PLAYER_1:
                    player1_count += 1
                else:
                    player2_count += 1

        return (player1_count - player2_count) / 19.0




class Tile:
    def __init__(self, id, neighbors, tileinfo) -> None:
        self.id = id
        self.neighbors = neighbors
        self.tileinfo = tileinfo


class TileInfo:
    # Słownik szablonów (zadeklarowany na końcu klasy, gdy klasa jest już w pełni widoczna)
    unit_template_library = {}
    data_len = 34 # 17


    def __init__(self, initiatives: list, melee_attacks: list, hp: int, player: Player, rot: int) -> None:
        # Konstruktor wymusza nową alokację list za pomocą [:] (płytka kopia na poziomie C)
        # Dzięki temu modyfikacja stanu na planszy nie zepsuje szablonu z biblioteki
        self.initiatives = initiatives[:]
        self.attacks = [0] * 6
        for i in range(len(melee_attacks)):
            self.attacks[(rot + i) % 6] = melee_attacks[i]

        self.alive = True
        self.hp = hp
        self.damage_taken = 0
        self.player = player

    def ensure_token(self):
        # print(f"sprawdzam token {self.damage_taken} : {self.hp}")
        if self.damage_taken >= self.hp:
            self.alive = False

    def copy_to(self, dest, rot) -> None:
        """
        Ultraszybkie przepisywanie stanu do innego, istniejącego już obiektu TileInfo (Zero nowej alokacji).
        Używamy [:] aby podmienić zawartość istniejących już list w pamięci.
        """
        dest.initiatives[:] = self.initiatives
        for i in range(len(self.attacks)):
            dest.attacks[(rot + i) % 6] = self.attacks[i]
        dest.alive = self.alive
        dest.hp = self.hp
        dest.damage_taken = self.damage_taken
        dest.player = self.player

    @staticmethod
    def load_to_np2(info: TileInfo, array):
        for i, val in enumerate(info.attacks):
            array[i] = val / 3.0
        for i, val in enumerate(info.initiatives):
            array[6 + i] = float(val)

        if info.player.value > 0:  # if player1 or player2
            array[12] = 1.0  # as owned
            array[13] = float(info.player.value - 1)  # we have 0 for player1, 1 for player 2
        else:
            array[12] = 0.0
            array[13] = 0.0
        array[14] = float(info.alive)
        array[15] = info.hp / 3.0
        array[16] = info.damage_taken / 3.0

    @staticmethod
    def load_to_np(info: TileInfo, array):
        offset = 0 if info.player == Player.PLAYER_1 else 17
        for i, val in enumerate(info.attacks):
            array[offset+ i] = val / 3.0
        for i, val in enumerate(info.initiatives):
            array[offset + 6 + i] = float(val)

        if info.player.value > 0:  # if player1 or player2
            array[offset + 12] = 1.0  # as owned
            array[offset + 13] = float(info.player.value - 1)  # we have 0 for player1, 1 for player 2
        else:
            array[offset + 12] = 0.0
            array[offset + 13] = 0.0
        array[offset + 14] = float(info.alive)
        array[offset + 15] = info.hp / 3.0
        array[offset + 16] = info.damage_taken / 3.0

    @classmethod
    def from_template(cls, template_key, player, rot):
        """Alternatywny konstruktor tworzący obiekt na bazie szablonu"""
        template = cls.unit_template_library[template_key]
        # Wywołuje domyślny __init__ przekazując odpowiednie dane

        instance = cls(template.initiatives, template.attacks, template.hp, player, rot)
        return instance

    @classmethod
    def insert_TileInfo(cls, tile, rot: int, player: Player, key: str) -> None:
        """
        Wstawia lub nadpisuje informacje o jednostce na kaflu.
        Jeśli kafel ma już zalokowaną pamięć, używa metody copy_to z szablonu.
        """
        template = cls.unit_template_library[key]

        if tile.tileinfo is None:
            # Brak obiektu -> Nowa alokacja (konstruktor zadba o skopiowanie list ze źródła)
            tile.tileinfo = cls.from_template(key, player, rot)

        else:
            # Obiekt istnieje -> Kopiujemy wartości z szablonu bezpośrednio do niego
            template.copy_to(tile.tileinfo, rot)
            # Unikalny stan gracza przypisujemy ręcznie po skopiowaniu bazy z szablonu
            tile.tileinfo.player = player
            tile.tileinfo.damage_taken = 0
            # tile.tileinfo.active = True
        tile.tileinfo.alive = True

    def __repr__(self) -> str:
        return (
            f"TileInfo("
            f"player={self.player.name}, "
            f"hp={self.hp}, "
            f"dmg={self.damage_taken}, "
            f"alive={self.alive}, "
            f"init={self.initiatives}, "
            f"atk={self.attacks}"
            f")"
        )


class HexBoard:

    def __init__(self) -> None:
        self.tiles = self._create_tiles()
        for tile in self.tiles:
            TileInfo.insert_TileInfo(tile, 0, Player.NONE, "none")
            tile.tileinfo.alive = False

    def _create_tiles(self):
        tiles = [
            Tile(0, [-1, -1, 2, 4, 1, -1], None),
            Tile(1, [-1, 0, 4, 6, 3, -1], None),
            Tile(2, [-1, -1, 5, 7, 4, 0], None),
            Tile(3, [-1, 1, 6, 8, -1, -1], None),
            Tile(4, [0, 2, 7, 9, 6, 1], None),
            Tile(5, [-1, -1, -1, 10, 7, 2], None),
            Tile(6, [1, 4, 9, 11, 8, 3], None),
            Tile(7, [2, 5, 10, 12, 9, 4], None),
            Tile(8, [3, 6, 11, 13, -1, -1], None),
            Tile(9, [4, 7, 12, 14, 11, 6], None),
            Tile(10, [5, -1, -1, 15, 12, 7], None),
            Tile(11, [6, 9, 14, 16, 13, 8], None),
            Tile(12, [7, 10, 15, 17, 14, 9], None),
            Tile(13, [8, 11, 16, -1, -1, -1], None),
            Tile(14, [9, 12, 17, 18, 16, 11], None),
            Tile(15, [10, -1, -1, -1, 17, 12], None),
            Tile(16, [11, 14, 18, -1, -1, 13], None),
            Tile(17, [12, 15, -1, -1, 18, 14], None),
            Tile(18, [14, 17, -1, -1, -1, 16], None),

        ]
        return tiles

    # --- INICJALIZACJA BIBLIOTEKI SZABLONÓW ---
    # W Pythonie definicję słownika odwołującego się do samej klasy robimy tuż pod jej ciałem
    TileInfo.unit_template_library = {
        "warrior": TileInfo([0, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0], 1, Player.NONE, 0),  # inicjatywa 2, 2 kierunki, 1hp
        "gladiator": TileInfo([0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 1, 0], 2, Player.NONE, 0),
        # inicjatywa 1, 4 kierunki, 2hp
        "assasin": TileInfo([0, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], 1, Player.NONE, 0),  # inicjatywa 3, 1 kierunek, 1hp
        "brawler": TileInfo([0, 1, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0], 1, Player.NONE, 0),
        # inicjatywy 2 i 1, 3 kierunki, 1hp
        "anti-swarm": TileInfo([0, 1, 0, 0, 0, 0], [1, 1, 1, 0, 1, 1], 1, Player.NONE, 0),
        # inicjatywy 2 i 1, 3 kierunki, 1hp
        "champion": TileInfo([1, 1, 0, 0, 0, 0], [2, 1, 0, 0, 0, 1], 1, Player.NONE, 0),
        # inicjatywy 2 i 1, 3 kierunki, 1hp
        "though_one": TileInfo([0, 1, 0, 0, 0, 0], [2, 0, 1, 0, 0, 0], 2, Player.NONE, 0),
        # inicjatywy 2 i 1, 3 kierunki, 1hp
        "HQ": TileInfo([1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], 20, Player.NONE, 0),
        "none": TileInfo([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], 0, Player.NONE, 0),

        # inicjatywy 2 i 1, 3 kierunki, 1hp
    }


class Drawer:
    def __init__(self) -> None:
        pass

    def get_hex(self) -> str:
        return fr"""
     _____
    /     \
   /       \
   \       /
    \_____/
    """

    def get_big_hex(self) -> str:
        return fr"""
      ________
     /        \
    /          \
   /            \
   \            /
    \          /
     \________/
    """

    def get_hex_from_tile(self, tileinfo: TileInfo) -> str:

        for i in range(5, 0, -1):
            if (tileinfo.initiatives[i] > 0):
                num = i
                break

        m = ['⚔' if i > 0 else ' ' for i in tileinfo.attacks]

        return fr"""
     _____
    /  {m[0]}  \
   /{m[5]}     {m[1]}\
   \{m[4]}  {num}  {m[2]}/
    \__{m[3]}__/
    """

    def get_hex2_from_tile(self, tileinfo: TileInfo) -> str:

        for k in range(5, -1, -1):
            if (tileinfo.initiatives[k] > 0):
                i = k
                break

        m = ['⚔' if i > 0 else ' ' for i in tileinfo.attacks]
        p = "P" + str(tileinfo.player.value)
        return fr"""
      ________
     /   {m[0]}    \
    /{m[5]}        {m[1]}\
   /     {p}     \
   \     {i}      /
    \{m[4]}        {m[2]}/
     \___{m[3]}____/
    """

    def get_board_template(self) -> str:
        return fr"""
                _____
               /     \
         _____/   0   \_____
        /     \       /     \
  _____/   1   \_____/   2   \_____
 /     \       /     \       /     \
/   3   \_____/   4   \_____/   5   \
\       /     \       /     \       /
 \_____/   6   \_____/   7   \_____/
 /     \       /     \       /     \
/   8   \_____/   9   \_____/  10   \
\       /     \       /     \       /
 \_____/  11   \_____/  12   \_____/
 /     \       /     \       /     \
/  13   \_____/  14   \_____/  15   \
\       /     \       /     \       /
 \_____/  16   \_____/  17   \_____/
       \       /     \       /
        \_____/  18   \_____/
              \       /
               \_____/
 """

    def get_board_template2(self) -> str:

        return fr"""
                         ________
                        /        \
                       /          \
              ________/     0      \________
             /        \            /        \
            /          \          /          \
   ________/     1      \________/     2      \________
  /        \            /        \            /        \
 /          \          /          \          /          \
/     3      \________/     4      \________/     5      \
\            /        \            /        \            /
 \          /          \          /          \          /
  \________/     6      \________/     7      \________/
  /        \            /        \            /        \
 /          \          /          \          /          \
/     8      \________/     9      \________/     10     \
\            /        \            /        \            /
 \          /          \          /          \          /
  \________/     11     \________/     12     \________/
  /        \            /        \            /        \
 /          \          /          \          /          \
/     13     \________/     14     \________/     15     \
\            /        \            /        \            /
 \          /          \          /          \          /
  \________/     16     \________/     17     \________/
           \            /        \            /
            \          /          \          /
             \________/     18     \________/
                      \            /
                       \          /
                        \________/
"""

    def get_board(self, board: HexBoard, config: dict = None) -> str:
        hide_player_none = True
        hide_dead = True
        print_debug_info = False

        if (config):
            hide_player_none = config.get("hide_player_none", True)
            hide_dead = config.get("hide_dead", True)
            print_debug_info = config.get("print_debug_info", False)

        ii = [None] * 19
        mm = [None] * 19
        pp = [None] * 19
        dd = [None] * 19
        sh = [None] * 19
        for n, item in enumerate(board.tiles):
            if print_debug_info:
                print(f"[{n}]: {item.tileinfo}")
            # Zabezpieczenie: Sprawdzamy czy item oraz item.tileinfo istnieją
            hide = ((item.tileinfo.player == Player.NONE and hide_player_none) or
                    (not item.tileinfo.alive and hide_dead))
            if item is None or item.tileinfo is None or hide:
                ii[n] = '  '
                mm[n] = [' '] * 6  # 6 pustych znaków dla każdej krawędzi ataku
                pp[n] = '  '  # dwa puste znaki, bo normalnie jest np. "P1"
                dd[n] = '  '
                sh[n] = '  '
                continue  # Przechodzimy do kolejnego kafelka

            # Jeśli tileinfo istnieje, wykonujemy standardową logikę:
            x = 0
            for k in range(6):
                if (item.tileinfo.initiatives[k] > 0):
                    if (x > 0):
                        ii[n] = str(k) + "*"
                    else:
                        ii[n] = str(k) + " "
                        x += 1

            if ii[n] is None:
                ii[n] = '--'
            mm[n] = ['x' if i > 0 else ' ' for i in item.tileinfo.attacks]
            color = '\x1b[37mP'
            if (item.tileinfo.player == Player.PLAYER_1):
                color = '\x1b[31mP'

            elif (item.tileinfo.player == Player.PLAYER_2):
                color = '\x1b[34mP'

            pp[n] = f"{color}{str(item.tileinfo.player.value)}\x1b[0m"

            dd[n] = '  ' if item.tileinfo.alive else '💀'
            sh[n] = '  ' if item.tileinfo.hp == 1 else '🛡'

        return fr"""
                             ________
                            /   {mm[0][0]}    \
                           /{mm[0][5]}        {mm[0][1]}\
                  ________/     {pp[0]}{dd[0]}   \________
                 /   {mm[1][0]}    \     {ii[0]}     /   {mm[2][0]}    \
                /{mm[1][5]}        {mm[1][1]}\{mm[0][4]}   {sh[0]}   {mm[0][2]}/{mm[2][5]}        {mm[2][1]}\
       ________/     {pp[1]}{dd[1]}   \___{mm[0][3]}____/     {pp[2]}{dd[2]}   \________
      /   {mm[3][0]}    \     {ii[1]}     /   {mm[4][0]}    \     {ii[2]}     /   {mm[5][0]}    \
     /{mm[3][5]}        {mm[3][1]}\{mm[1][4]}   {sh[1]}   {mm[1][2]}/{mm[4][5]}        {mm[4][1]}\{mm[2][4]}   {sh[2]}   {mm[2][2]}/{mm[5][5]}        {mm[5][1]}\
    /     {pp[3]}{dd[3]}   \___{mm[1][3]}____/     {pp[4]}{dd[4]}   \___{mm[2][3]}____/     {pp[5]}{dd[5]}   \
    \     {ii[3]}     /   {mm[6][0]}    \     {ii[4]}     /   {mm[7][0]}    \     {ii[5]}     /
     \{mm[3][4]}   {sh[3]}   {mm[3][2]}/{mm[6][5]}        {mm[6][1]}\{mm[4][4]}   {sh[4]}   {mm[4][2]}/{mm[7][5]}        {mm[7][1]}\{mm[5][4]}   {sh[5]}   {mm[5][2]}/
      \___{mm[3][3]}____/     {pp[6]}{dd[6]}   \___{mm[4][3]}____/     {pp[7]}{dd[7]}   \___{mm[5][3]}____/
      /   {mm[8][0]}    \     {ii[6]}     /   {mm[9][0]}    \     {ii[7]}     /   {mm[10][0]}    \
     /{mm[8][5]}        {mm[8][1]}\{mm[6][4]}   {sh[6]}   {mm[6][2]}/{mm[9][5]}        {mm[9][1]}\{mm[7][4]}   {sh[7]}   {mm[7][2]}/{mm[10][5]}        {mm[10][1]}\
    /     {pp[8]}{dd[8]}   \___{mm[6][3]}____/     {pp[9]}{dd[9]}   \___{mm[7][3]}____/     {pp[10]}{dd[10]}   \
    \     {ii[8]}     /   {mm[11][0]}    \     {ii[9]}     /   {mm[12][0]}    \     {ii[10]}     /
     \{mm[8][4]}   {sh[8]}   {mm[8][2]}/{mm[11][5]}        {mm[11][1]}\{mm[9][4]}   {sh[9]}   {mm[9][2]}/{mm[12][5]}        {mm[12][1]}\{mm[10][4]}   {sh[10]}   {mm[10][2]}/
      \___{mm[8][3]}____/     {pp[11]}{dd[11]}   \___{mm[9][3]}____/     {pp[12]}{dd[12]}   \___{mm[10][3]}____/
      /   {mm[13][0]}    \     {ii[11]}     /   {mm[14][0]}    \     {ii[12]}     /   {mm[15][0]}    \
     /{mm[13][5]}        {mm[13][1]}\{mm[11][4]}   {sh[11]}   {mm[11][2]}/{mm[14][5]}        {mm[14][1]}\{mm[12][4]}   {sh[12]}   {mm[12][2]}/{mm[15][5]}        {mm[15][1]}\
    /     {pp[13]}{dd[13]}   \___{mm[11][3]}____/     {pp[14]}{dd[14]}   \___{mm[12][3]}____/     {pp[15]}{dd[15]}   \
    \     {ii[13]}     /   {mm[16][0]}    \     {ii[14]}     /   {mm[17][0]}    \     {ii[15]}     /
     \{mm[13][4]}   {sh[13]}   {mm[13][2]}/{mm[16][5]}        {mm[16][1]}\{mm[14][4]}   {sh[14]}   {mm[14][2]}/{mm[17][5]}        {mm[17][1]}\{mm[15][4]}   {sh[15]}   {mm[15][2]}/
      \___{mm[13][3]}____/     {pp[16]}{dd[16]}   \___{mm[14][3]}____/     {pp[17]}{dd[17]}   \___{mm[15][3]}____/
               \     {ii[16]}     /   {mm[18][0]}    \     {ii[17]}     /
                \{mm[16][4]}   {sh[16]}   {mm[16][2]}/{mm[18][5]}        {mm[18][1]}\{mm[17][4]}   {sh[17]}   {mm[17][2]}/
                 \___{mm[16][3]}____/     {pp[18]}{dd[18]}   \___{mm[17][3]}____/
                          \     {ii[18]}     /
                           \{mm[18][4]}   {sh[18]}   {mm[18][2]}/
                            \___{mm[18][3]}____/
    """


#drawer = Drawer()
# board = HexBoard()
# for tile in board.tiles:
#     TileInfo.insert_TileInfo(tile,0,Player.PLAYER_1,"warrior")


# state = GameState()
# # TileInfo.insert_TileInfo(state.board.tiles[18], 1, Player.PLAYER_1, "anti-swarm")
# # board.tiles[18].tileinfo.alive = False
# # TileInfo.insert_TileInfo(state.board.tiles[17], 3, Player.PLAYER_2, "warrior")
# state.randomly_populate()
# print(drawer.get_board(state.board, {"hide_player_none": True}))
# state.battle()
# # print(drawer.get_board_template2())
# # print(drawer.get_board_template())
# # print(drawer.get_hex())
# # print(drawer.get_hex_from_tile(TileInfo.unit_template_library["warrior"]))
# # print(drawer.get_hex2_from_tile(TileInfo.unit_template_library["warrior"]))
# print(drawer.get_board(state.board, {"hide_player_none": True, "print_debug_info": True}))

drawer = Drawer()
state = GameState()

player, hand, result = state.reset()
print(f"{player=}, {hand=}, {result=}")



def get_int_input(prompt, min_val, max_val, allow_skip=False):
    """Pomocnicza funkcja do bezpiecznego pobierania liczb z konsoli."""
    while True:
        user_input = input(prompt).strip()

        if allow_skip and user_input.lower() == "s":
            return -1

        try:
            val = int(user_input)
            if min_val <= val <= max_val:
                return val
            print(f"❌ Błąd: Podaj liczbę z zakresu od {min_val} do {max_val}.")
        except ValueError:
            if allow_skip:
                print("❌ Błąd: Podaj liczbę lub wpisz 's' aby pominąć.")
            else:
                print("❌ Błąd: To nie jest poprawna liczba.")


def get_placement_input(placement_name):
    """Pobiera pełne dane dla jednego placementu (Wystawienia)."""
    print(f"\n--- Konfiguracja {placement_name} ---")
    print("Wpisz 's' na dowolnym etapie, aby pominąć ten placement.")

    hand_idx = get_int_input(
        "-> Podaj indeks żetonu z ręki (0-2) lub 's': ",
        0,
        2,
        allow_skip=True,
    )
    if hand_idx == -1:
        return None  # Gracz pomija ten placement

    board_idx = get_int_input(
        "-> Podaj indeks pola na planszy (0-18): ", 0, 18
    )
    rot = get_int_input("-> Podaj rotację żetonu (0-5): ", 0, 5)

    return {"hand_idx": hand_idx, "board_idx": board_idx, "rot": rot}

def play():
    drawed = []
    while not result:
        for item in drawed:
            print(item)

        print(f"{player=}")
        k = 0
        for i, token in enumerate(hand):
            if token == None:
                continue
            k += 1
            print(f"\n\n----------token <{i}> ---------------\n\n")
            if token in state.action_tokens:
                print(f"token akcji: {token}")
            else:
                info = TileInfo.unit_template_library[token]
                print(drawer.get_hex2_from_tile(info))
                print(f"token jednostki: {info}")

        while True:


            print("\n==========================================")
            print("      NOWA AKCJA UŻYTKOWNIKA (TURA)       ")
            print("==========================================")

            # 1. Pobranie żetonu do odrzucenia (Discard)
            print("Krok 1: Odrzucenie żetonu (Każda tura wymaga odrzucenia min. 1).")
            discard_idx = 2
            if k == 3:
                discard_idx = get_int_input(
                    "Podaj indeks żetonu z ręki do odrzucenia (0-2): ", 0, 2
                )

            # 2. Pobranie pierwszego placementu
            placement1 = get_placement_input("Placement 1")

            # 3. Pobranie drugiego placementu (tylko jeśli pierwszy został uzupełniony,
            # w Neuroshimie nie można zrobić Placement 2, pomijając Placement 1)
            placement2 = None
            if placement1 is not None:
                placement2 = get_placement_input("Placement 2")
            else:
                print("\nSkipping Placement 2 (Placement 1 was empty).")

            # 4. Złożenie struktury GameAction
            # Jeśli placementy zostały pominięte, uzupełniamy je wartościami -1 (zgodnie z logiką C#)
            game_action = {
                "discard_idx": discard_idx,
                "placement1": placement1
                if placement1
                else {"hand_idx": -1, "board_idx": -1, "rot": -1},
                "placement2": placement2
                if placement2
                else {"hand_idx": -1, "board_idx": -1, "rot": -1},
            }



            # Opcja wyjścia z pętli testowej
            cont = input("\nCzy chcesz sparsować kolejną akcję? (t/n): ").strip().lower()
            if cont == "n":
                print("Koniec pracy parsera.")
                break


        player, hand, result, drawed = state.step(game_action, config={"draw": True})