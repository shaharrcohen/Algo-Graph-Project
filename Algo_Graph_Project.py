from collections import Counter, deque, defaultdict
from typing import Dict, List, Tuple, Set, Any
import random, string, time

# ============================================================
#                    QUESTION 1 — SECTION A
#                 (Huffman encoding: encode_text)
# ============================================================

# 1) Unicode-lexicographic, deterministic radix sort helpers
def _radix_sort_symbols_lex(symbols: List[str]) -> List[str]:
    """
    Deterministic, stable lexicographic sort by Unicode code points.
    LSD Radix with base=256 and 3 fixed passes (ord <= 2^21 - 1).
    Time: O(s) with a small constant (3 passes); Space: O(s).
    """
    if not symbols:
        return []
    codes = [ord(ch) for ch in symbols]
    for shift in (0, 8, 16):  # 3 passes cover up to 21 bits
        buckets = [[] for _ in range(256)]
        for i, cp in enumerate(codes):
            buckets[(cp >> shift) & 0xFF].append(i)
        new_symbols, new_codes = [], []
        for bucket in buckets:
            for idx in bucket:
                new_symbols.append(symbols[idx])
                new_codes.append(codes[idx])
        symbols, codes = new_symbols, new_codes
    return symbols


def _order_symbols_by_freq_then_lex(freq: Dict[str, int], n: int) -> List[str]:
    """
    Return symbols ordered by decreasing frequency; ties broken lexicographically.
    Linear time via counting buckets on frequencies (0..n) and radix within buckets.
    """
    if not freq:
        return []
    buckets: List[List[str]] = [[] for _ in range(n + 1)]
    for ch, f in freq.items():
        buckets[f].append(ch)
    ordered: List[str] = []
    for f in range(n, 0, -1):  # high freq -> low
        if buckets[f]:
            ordered.extend(_radix_sort_symbols_lex(buckets[f]))
    return ordered


# A1 — encode_text (SPEC)
def encode_text(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Returns (encoded_text, codes) according to the specification with linear-time
    construction (Θ(n + s + L)):
      - Optimal Huffman coding (two-queue method) to compute code lengths.
      - Single-character case: '0' and all zeros.
      - Empty string case: ("", {}).
      - Canonical Huffman codes (deterministic).
      - Returned dict insertion order: decreasing frequency; ties -> lexicographic.
    """
    # Empty string
    if not text:
        return "", {}

    freq = Counter(text)
    s = len(freq)

    # Single unique symbol
    if s == 1:
        ch = next(iter(freq))
        return "0" * len(text), {ch: "0"}

    n = len(text)

    # ---------- Two-queue Huffman build ----------
    # Prepare leaves buckets by frequency; within each bucket, lex-sort via radix.
    leaves_by_freq: List[List[str]] = [[] for _ in range(n + 1)]
    for ch, f in freq.items():
        leaves_by_freq[f].append(ch)
    for f in range(1, n + 1):
        if leaves_by_freq[f]:
            leaves_by_freq[f] = _radix_sort_symbols_lex(leaves_by_freq[f])

    # q1: leaves in nondecreasing freq; q2: internal nodes
    q1: deque[Tuple[int, Any]] = deque()
    for f in range(1, n + 1):
        for ch in leaves_by_freq[f]:
            q1.append((f, ch))
    q2: deque[Tuple[int, Any]] = deque()

    def _pop_min() -> Tuple[int, Any]:
        if q1 and q2:
            return q1.popleft() if q1[0][0] <= q2[0][0] else q2.popleft()
        elif q1:
            return q1.popleft()
        else:
            return q2.popleft()

    while True:
        f1, n1 = _pop_min()
        f2, n2 = _pop_min()
        merged = (n1, n2)
        q2.append((f1 + f2, merged))
        if not q1 and len(q2) == 1:
            break

    _, root = q2[0]

    # ---------- Compute code lengths (iterative DFS) ----------
    code_lengths: Dict[str, int] = {}
    stack: List[Tuple[Any, int]] = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if isinstance(node, str):
            code_lengths[node] = depth
        else:
            left, right = node
            stack.append((right, depth + 1))
            stack.append((left, depth + 1))

    # ---------- Canonical Huffman codes (linear sorting by (length, symbol)) ----------
    max_len = max(code_lengths.values()) if code_lengths else 0
    len_buckets: List[List[str]] = [[] for _ in range(max_len + 1)]
    for ch, L in code_lengths.items():
        len_buckets[L].append(ch)
    for L in range(len(len_buckets)):
        if len_buckets[L]:
            len_buckets[L] = _radix_sort_symbols_lex(len_buckets[L])

    codes_tmp: Dict[str, str] = {}
    code_val = 0
    prev_len = 0
    for L in range(len(len_buckets)):
        if not len_buckets[L]:
            continue
        if L != prev_len:
            code_val <<= (L - prev_len)
            prev_len = L
        for ch in len_buckets[L]:
            codes_tmp[ch] = format(code_val, f"0{L}b") if L > 0 else ""
            code_val += 1

    # ---------- Return dict order by freq↓ then lex ----------
    ordered_symbols = _order_symbols_by_freq_then_lex(freq, n)
    codes: Dict[str, str] = {ch: codes_tmp[ch] for ch in ordered_symbols}

    # ---------- Encode ----------
    encoded = "".join(codes[ch] for ch in text)
    return encoded, codes


# ============================================================
#           (Q1 helper checks used in main for A)
# ============================================================

class _TrieNode:
    __slots__ = ("children", "terminal", "sym")

    def __init__(self):
        self.children: Dict[str, "_TrieNode"] = {}
        self.terminal: bool = False
        self.sym: str | None = None


def _is_prefix_free(codes: Dict[str, str]) -> bool:
    root = _TrieNode()
    for sym, code in codes.items():
        node = root
        for bit in code:
            if node.terminal:
                return False
            node = node.children.setdefault(bit, _TrieNode())
        if node.children or node.terminal:
            return False
        node.terminal = True
        node.sym = sym
    return True


def _canonical_ok(codes: Dict[str, str]) -> bool:
    items = sorted(codes.items(), key=lambda kv: (len(kv[1]), kv[0]))
    if not items:
        return True
    prev_len = len(items[0][1])
    prev_val = int(items[0][1] or "0", 2) if prev_len > 0 else 0
    for ch, code in items[1:]:
        L = len(code)
        val = int(code or "0", 2) if L > 0 else 0
        if L == prev_len:
            if val != prev_val + 1:
                return False
        else:
            d = L - prev_len
            if val != (prev_val + 1) << d:
                return False
        prev_len, prev_val = L, val
    return True


def _dict_order_ok(text: str, codes: Dict[str, str]) -> bool:
    freqs = Counter(text)
    expected = [k for k, _ in sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))]
    return expected == list(codes.keys())


def _decode_for_test(encoded: str, codes: Dict[str, str]) -> str:
    root = _TrieNode()
    for ch, code in codes.items():
        node = root
        for b in code:
            node = node.children.setdefault(b, _TrieNode())
        node.terminal = True
        node.sym = ch
    out: List[str] = []
    node = root
    for b in encoded:
        if b not in node.children:
            raise ValueError("Decoding error")
        node = node.children[b]
        if node.terminal:
            out.append(node.sym)  # type: ignore[arg-type]
            node = root
    if node is not root:
        raise ValueError("Incomplete code")
    return "".join(out)


def _build_tests() -> List[Tuple[str, str]]:
    tests: List[Tuple[str, str]] = []
    # Edge cases
    tests.append(("empty", ""))
    tests.append(("single-1", "a"))
    tests.append(("single-10", "a" * 10))
    # Two symbols
    tests.append(("two-imbalanced", "aaaaab"))
    tests.append(("two-equal", "abababab"))
    # Equal-ish distributions
    tests.append(("three-equal", "abc"))
    tests.append(("four-equalish", "abca" * 5))
    # Spaces & punctuation
    tests.append(("pangram", "the quick brown fox jumps over the lazy dog"))
    # Hebrew + punctuation (Unicode)
    tests.append(("hebrew", "אבגד אב! אב?"))
    # Random stress
    random.seed(17)

    def r(n: int, alphabet: str) -> str:
        return "".join(random.choice(alphabet) for _ in range(n))

    tests.append(("rand-2-2e4", r(20000, "01")))
    tests.append(("rand-10-1e4", r(10000, string.ascii_lowercase[:10])))
    tests.append(("rand-100-5e3", r(5000, string.ascii_letters[:100])))
    # Pathological: many unique symbols once each (s=n)
    alphabet = "".join(chr(i) for i in range(65, 65 + 80))
    tests.append(("many-unique", "".join(alphabet)))
    return tests


# ============================================================
#                    QUESTION 1 — SECTION B
#                 (Huffman decoding: decode_text)
# ============================================================

class HuffmanTrieNode:
    def __init__(self):
        self.children: Dict[str, "HuffmanTrieNode"] = {}
        self.char: str | None = None


def build_trie(codes: Dict[str, str]) -> HuffmanTrieNode:
    """
    Build a prefix-free Huffman Trie from a dict mapping characters to codes.
    Validates:
      - codes is a dict of str->str
      - codes contain only '0'/'1'
      - no empty codeword
      - no duplicate codeword
      - no code is a prefix of another code (prefix-free)
    Raises ValueError / TypeError accordingly.
    """
    if not isinstance(codes, dict):
        raise TypeError("codes must be a dictionary.")
    for k, v in codes.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise TypeError("All keys and values in codes must be strings.")
        if any(b not in ("0", "1") for b in v):
            raise ValueError("All codes must contain only '0' and '1'.")
        if v == "":
            raise ValueError(f"Invalid code for '{k}': empty codeword is not allowed.")

    root = HuffmanTrieNode()
    # Insert shorter codes first for clearer error messages
    for char, code in sorted(codes.items(), key=lambda kv: (len(kv[1]), kv[0])):
        node = root
        for i, bit in enumerate(code):
            if bit not in node.children:
                node.children[bit] = HuffmanTrieNode()
            node = node.children[bit]
            # If a code already ends here and we haven't finished this code -> prefix violation
            if node.char is not None and i < len(code) - 1:
                raise ValueError(
                    f"Provided codes are not prefix-free: existing code for '{node.char}' "
                    f"is a prefix of the code for '{char}'."
                )
        # Ending on an already-used leaf -> duplicate
        if node.char is not None:
            raise ValueError(
                f"Duplicate code detected: '{code}' is assigned to both '{node.char}' and '{char}'."
            )
        # If this node already has children -> new code is a prefix of some existing code
        if node.children:
            raise ValueError(
                f"Provided codes are not prefix-free: code for '{char}' is a prefix of another code."
            )
        node.char = char

    return root


def decode_text(encoded_text: str, codes: Dict[str, str]) -> str:
    """
    Decode a binary string into the original text using a Huffman Trie.

    Time complexity:
      - Trie build:  O(m * L_avg), where m = #codes, L_avg = avg code length
      - Decoding:    O(n), where n = number of bits in encoded_text
    """
    if not isinstance(encoded_text, str):
        raise TypeError("encoded_text must be a string.")
    if any(b not in ("0", "1") for b in encoded_text):
        raise ValueError("encoded_text must contain only '0' and '1'.")

    # Empty text is trivially decodable (even with empty codes)
    if encoded_text == "":
        return ""

    root = build_trie(codes)
    if not codes:
        # Non-empty text without codes cannot be decoded
        raise ValueError("Encoded text cannot be decoded with an empty codes dictionary.")

    out: List[str] = []
    node = root
    for i, bit in enumerate(encoded_text):
        if bit not in node.children:
            raise ValueError(f"Invalid bit sequence at position {i}: no path for bit '{bit}'.")
        node = node.children[bit]
        if node.char is not None:
            out.append(node.char)
            node = root

    if node is not root:
        raise ValueError("Encoded text ends with an incomplete codeword.")

    return "".join(out)


# Rigorous Edge-Case Tests for Q1B (decoding)
def _run_case(name: str, encoded: Any, codes: Any, expect: Any = None, should_pass: bool = True) -> dict:
    """
    Run a single test case. Returns a dict row with results.
    """
    try:
        out = decode_text(encoded, codes)
        passed = should_pass and (expect is None or out == expect)
        note = f"decoded='{out}'"
        exc = ""
    except Exception as e:
        out = None
        passed = (not should_pass)  # if we expected failure, exception means pass
        note = ""
        exc = f"{type(e).__name__}: {str(e)}"

    return {
        "case": name,
        "should_pass": should_pass,
        "passed": passed,
        "encoded_len": (len(encoded) if isinstance(encoded, str) else None),
        "num_codes": (len(codes) if isinstance(codes, dict) else None),
        "expect": expect,
        "note": note,
        "exception": exc
    }


def run_strict_suite() -> List[dict]:
    rows: List[dict] = []

    # --- Passing (valid) ---
    rows.append(_run_case("normal-basic",
                          "101000011",
                          {"b": "0", "a": "10", "c": "11"},
                          "aabbbc", True))

    rows.append(_run_case("single-symbol",
                          "0000",
                          {"z": "0"},
                          "zzzz", True))

    rows.append(_run_case("empty-text_with_codes",
                          "", {"a": "0"},
                          "", True))

    rows.append(_run_case("empty-text_empty-codes",
                          "", {},
                          "", True))

    rows.append(_run_case("multi-len-codes",
                          "1110110",
                          {"x": "111", "y": "0", "z": "110"},
                          "xyz", True))

    rows.append(_run_case("unicode-hebrew",
                          "01011",
                          {"א": "0", "ב": "10", "ג": "11"},
                          "אבג", True))

    rows.append(_run_case("long-repeat",
                          "0" * 2000,
                          {"x": "0"},
                          "x" * 2000, True))

    # --- Failing / Validation ---
    rows.append(_run_case("non-binary-in-encoded",
                          "10a01",
                          {"a": "0", "b": "10"},
                          None, False))

    rows.append(_run_case("non-binary-in-codes",
                          "1010",
                          {"a": "0", "b": "1x"},
                          None, False))

    rows.append(_run_case("empty-codeword_in_codes",
                          "0",
                          {"a": "", "b": "0"},
                          None, False))

    rows.append(_run_case("non-prefix-free_1",
                          "011",
                          {"a": "0", "b": "01"},
                          None, False))  # '0' is prefix of '01'

    rows.append(_run_case("non-prefix-free_2",
                          "010",
                          {"a": "0", "b": "00"},
                          None, False))  # '0' is prefix of '00'

    rows.append(_run_case("duplicate-code",
                          "0",
                          {"a": "0", "b": "0"},
                          None, False))

    rows.append(_run_case("empty-codes_nonempty-text",
                          "101",
                          {},
                          None, False))

    rows.append(_run_case("unknown-path",
                          "1111",
                          {"a": "0", "b": "10"},
                          None, False))

    rows.append(_run_case("incomplete-at-end",
                          "1",
                          {"a": "10", "b": "0"},
                          None, False))

    # --- Type errors ---
    rows.append(_run_case("bad-types_encoded_not_str",
                          10101, {"a": "0"},
                          None, False))

    rows.append(_run_case("bad-types_codes_not_dict",
                          "1010", [("a", "0")],
                          None, False))

    rows.append(_run_case("bad-types_value_not_str",
                          "1010", {"a": 0},
                          None, False))

    rows.append(_run_case("bad-types_key_not_str",
                          "1010", {1: "0"},
                          None, False))

    return rows


def _print_table(rows: List[dict]) -> None:
    # Minimal columnar print; avoids external deps.
    cols = ["case", "should_pass", "passed", "encoded_len", "num_codes", "expect", "note", "exception"]
    widths = {c: max(len(c), max(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    sep = " | "

    # Header
    header = sep.join(c.ljust(widths[c]) for c in cols)
    line = "-+-".join("-" * widths[c] for c in cols)
    print(header)
    print(line)

    # Rows
    for r in rows:
        print(sep.join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


# ============================================================
#                    QUESTION 2 — SECTION A
#               (Tree isomorphism: are_isomorphic)
# ============================================================

# ---------- Types for Q2 ----------
Matrix = List[List[bool]]
Adj = List[List[int]]
Mapping = List[Tuple[int, int]]

# Core utilities (matrix_to_adj, is_tree, find_centers, root_tree)
def matrix_to_adj(M: Matrix) -> Adj:
    """
    Convert an undirected adjacency matrix (bool/0-1) to adjacency lists.
    Validates: square, symmetric, zero diagonal.
    Time: O(n^2) due to the input format.
    """
    n = len(M)
    if any(len(row) != n for row in M):
        raise ValueError("Adjacency matrix must be square.")

    for i in range(n):
        if M[i][i]:
            raise ValueError("Self-loop not allowed on the diagonal.")
        for j in range(i + 1, n):
            if bool(M[i][j]) != bool(M[j][i]):
                raise ValueError("Matrix must be symmetric for an undirected graph.")

    adj: Adj = [[] for _ in range(n)]
    for i in range(n):
        for j, val in enumerate(M[i]):
            if val:
                adj[i].append(j)
    return adj


def is_tree(adj: Adj) -> bool:
    """
    Verify the graph is a single tree:
    - connected
    - edges = n-1
    Time: O(n)
    """
    n = len(adj)
    if n == 0:
        return True
    edges = sum(len(nei) for nei in adj) // 2
    if edges != n - 1:
        return False

    seen = [False] * n
    q = deque([0])
    seen[0] = True
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                q.append(v)
    return all(seen)


def find_centers(adj: Adj) -> List[int]:
    """
    Find 1 or 2 centers of a tree by iterative leaf-stripping. O(n).
    """
    n = len(adj)
    if n == 0:
        return []
    if n == 1:
        return [0]

    deg = [len(adj[i]) for i in range(n)]
    leaves = deque([i for i in range(n) if deg[i] <= 1])
    remaining = n

    while remaining > 2:
        sz = len(leaves)
        remaining -= sz
        for _ in range(sz):
            leaf = leaves.popleft()
            deg[leaf] = 0
            for nb in adj[leaf]:
                if deg[nb] > 0:
                    deg[nb] -= 1
                    if deg[nb] == 1:
                        leaves.append(nb)
    return list(leaves)


def root_tree(adj: Adj, root: int) -> Tuple[List[int], List[List[int]]]:
    """
    Root the tree at 'root' and return (parent, children_list).
    Time: O(n)
    """
    n = len(adj)
    parent = [-1] * n
    children = [[] for _ in range(n)]
    stack = [root]
    parent[root] = root

    while stack:
        u = stack.pop()
        for v in adj[u]:
            if parent[v] == -1:
                parent[v] = u
                children[u].append(v)
                stack.append(v)

    return parent, children


# Canonical coding with GLOBAL interning (O(n), no sorting)
class CanonicalIDEncoder:
    """
    Global interning of subtree-signature multisets -> stable small integer IDs.
    Key idea for O(n): represent each node's children signature multiset as a
    frozenset of (child_sig_id, multiplicity) pairs (order-free, hashable).
    No per-node sorting is required.
    """
    def __init__(self):
        self.sig_map: Dict[frozenset, int] = {}
        self.next_id: int = 1  # IDs start at 1; leaf (empty multiset) will get 1.

    def encode_tree(self, adj: Adj, root: int) -> Tuple[int, List[int]]:
        """
        Encode a rooted tree and return (root_id, id_per_node) using global interning.
        Time: O(n) across the tree (sum of degrees = 2(n-1)).
        """
        _, children = root_tree(adj, root)

        # postorder traversal
        order = []
        stack = [(root, 0)]
        while stack:
            u, st = stack.pop()
            if st == 0:
                stack.append((u, 1))
                for v in children[u]:
                    stack.append((v, 0))
            else:
                order.append(u)

        sig: List[int] = [0] * len(adj)
        for u in order:
            # Count child signature IDs (no sorting)
            cnt = Counter(sig[v] for v in children[u])
            key = frozenset(cnt.items())  # order-insensitive, hashable
            if key not in self.sig_map:
                self.sig_map[key] = self.next_id
                self.next_id += 1
            sig[u] = self.sig_map[key]

        return sig[root], sig


# Public API (Q2A)
def are_isomorphic(tree_1: Matrix, tree_2: Matrix) -> bool:
    """
    Return True iff two (undirected, unlabelled) input trees are isomorphic.
    Uses GLOBAL-ID AHU coding from centers with a shared encoder (canonical across trees).
    For n=0 (both empty), returns True by convention.
    Overall time dominated by reading the matrix: O(n^2) + O(n).
    """
    n1, n2 = len(tree_1), len(tree_2)
    if n1 != n2:
        return False
    n = n1
    if n == 0:
        return True

    adj1 = matrix_to_adj(tree_1)
    adj2 = matrix_to_adj(tree_2)

    if not is_tree(adj1) or not is_tree(adj2):
        return False

    centers1 = find_centers(adj1)
    centers2 = find_centers(adj2)

    enc = CanonicalIDEncoder()  # shared across trees ⇒ consistent IDs

    root_codes_1 = set()
    for c in centers1:
        code_c, _ = enc.encode_tree(adj1, c)
        root_codes_1.add(code_c)

    for c in centers2:
        code_c, _ = enc.encode_tree(adj2, c)
        if code_c in root_codes_1:
            return True

    return False


# ============================================================
#                    QUESTION 2 — SECTION B
#          (Tree isomorphism mapping: mapping_isomorphic)
# ============================================================

def validate_isomorphism(adj1: Adj, adj2: Adj, mapping: List[Tuple[int, int]]) -> bool:
    """
    Validate that 'mapping' is a bijection V(T1)->V(T2) preserving adjacency.
    Time: O(n + m) = O(n) for trees.
    """
    n1 = len(adj1)
    n2 = len(adj2)
    if n1 != n2:
        return False
    n = n1

    f = [-1] * n
    seen = [False] * n
    for v, w in mapping:
        if not (0 <= v < n and 0 <= w < n):
            return False
        if f[v] != -1 or seen[w]:
            return False
        f[v] = w
        seen[w] = True
    if any(x == -1 for x in f):
        return False

    adj2_sets = [set(nei) for nei in adj2]
    for u in range(n):
        fu = f[u]
        for v in adj1[u]:
            if f[v] not in adj2_sets[fu]:
                return False
    return True


def mapping_isomorphic(tree_1: Matrix, tree_2: Matrix) -> List[Tuple[int, int]]:
    """
    If trees are isomorphic, return a concrete node mapping as a list of pairs (v, w),
    otherwise return [].
    Uses GLOBAL-ID AHU coding (O(n)) and validates adjacency preservation.
    """
    n1, n2 = len(tree_1), len(tree_2)
    if n1 != n2:
        return []
    n = n1
    if n == 0:
        # No vertices to map; per assignment we'll return [] for 2B.
        return []

    adj1 = matrix_to_adj(tree_1)
    adj2 = matrix_to_adj(tree_2)

    if not is_tree(adj1) or not is_tree(adj2):
        return []

    centers1 = find_centers(adj1)
    centers2 = find_centers(adj2)

    enc = CanonicalIDEncoder()

    # simple caches to avoid recomputation per center
    root_cache_1: Dict[int, Tuple[List[int], List[List[int]]]] = {}
    code_cache_1: Dict[int, Tuple[int, List[int]]] = {}
    root_cache_2: Dict[int, Tuple[List[int], List[List[int]]]] = {}
    code_cache_2: Dict[int, Tuple[int, List[int]]] = {}

    def get_rooting(adj: Adj, c: int, cache: Dict[int, Tuple[List[int], List[List[int]]]]) -> Tuple[List[int], List[List[int]]]:
        if c not in cache:
            cache[c] = root_tree(adj, c)
        return cache[c]

    def get_codes(adj: Adj, c: int, cache: Dict[int, Tuple[int, List[int]]]) -> Tuple[int, List[int]]:
        if c not in cache:
            cache[c] = enc.encode_tree(adj, c)
        return cache[c]

    for c1 in centers1:
        root_code_1, sig1 = get_codes(adj1, c1, code_cache_1)
        _, ch1 = get_rooting(adj1, c1, root_cache_1)

        for c2 in centers2:
            root_code_2, sig2 = get_codes(adj2, c2, code_cache_2)
            if root_code_1 != root_code_2:
                continue

            _, ch2 = get_rooting(adj2, c2, root_cache_2)
            mapping_pairs: List[Tuple[int, int]] = []

            def dfs(u: int, v: int) -> bool:
                if sig1[u] != sig2[v]:
                    return False
                mapping_pairs.append((u, v))

                # Group children by signature ID; no sorting needed except stable output order
                buckets1: Dict[int, List[int]] = defaultdict(list)
                buckets2: Dict[int, List[int]] = defaultdict(list)
                for cu in ch1[u]:
                    buckets1[sig1[cu]].append(cu)
                for cv in ch2[v]:
                    buckets2[sig2[cv]].append(cv)

                if set(buckets1.keys()) != set(buckets2.keys()):
                    return False

                for s in buckets1.keys():
                    L1 = buckets1[s]
                    L2 = buckets2[s]
                    if len(L1) != len(L2):
                        return False
                    # pair in any consistent order; choose index order for determinism
                    L1_sorted = sorted(L1)
                    L2_sorted = sorted(L2)
                    for a, b in zip(L1_sorted, L2_sorted):
                        if not dfs(a, b):
                            return False
                return True

            if dfs(c1, c2):
                if len(mapping_pairs) == n and validate_isomorphism(adj1, adj2, mapping_pairs):
                    return sorted(mapping_pairs, key=lambda p: p[0])

    return []


# ------------------------------------------------------------
# Q2 helpers for tests
# ------------------------------------------------------------
def mat_from_edges(n: int, edges: List[Tuple[int, int]]) -> Matrix:
    """
    Build an adjacency matrix from an undirected edge list.
    Raises on self-loops. Time: O(n + m) to build the matrix.
    """
    M = [[False] * n for _ in range(n)]
    for u, v in edges:
        if u == v:
            raise ValueError("No self-loops allowed in a tree.")
        M[u][v] = True
        M[v][u] = True
    return M

def print_mapping(mapping: List[Tuple[int, int]]) -> None:
    """Pretty-print a mapping for manual inspection."""
    if not mapping:
        print("Mapping: [] (not isomorphic)")
    else:
        print("Mapping (v in T1 -> w in T2):")
        for v, w in sorted(mapping):
            print(f"  {v} -> {w}")

def permute_edges(n: int, edges: List[Tuple[int, int]], perm: List[int]) -> List[Tuple[int, int]]:
    """Apply a node-label permutation 'perm' to an undirected edge list."""
    return [(perm[u], perm[v]) for u, v in edges]

def edges_from_matrix(M: Matrix) -> List[Tuple[int, int]]:
    """Extract undirected edge list (i<j) from adjacency matrix."""
    n = len(M)
    E: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if M[i][j]:
                E.append((i, j))
    return E

def edges_canonical_from_matrix(M: Matrix) -> Set[Tuple[int, int]]:
    """Collect undirected edges (i<j) from adjacency matrix into a set."""
    n = len(M)
    S: Set[Tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            if M[i][j]:
                S.add((i, j))
    return S

def edges_apply_mapping(n: int, edges: List[Tuple[int, int]], mapping: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """
    Apply node mapping to an undirected edge list and return canonical set of edges (i<j).
    """
    f = [-1] * n
    for v, w in mapping:
        f[v] = w
    if any(x == -1 for x in f):
        raise AssertionError("Mapping is not a full bijection on V1.")
    out: Set[Tuple[int, int]] = set()
    for u, v in edges:
        a, b = f[u], f[v]
        if a > b:
            a, b = b, a
        out.add((a, b))
    return out


# ============================================================
#                          UNIFIED MAIN
# ============================================================

def main():
    # --------------------------
    # Q1A: self-check & benchmark (encoding)
    # --------------------------
    print("=== Q1A: encode_text — self-check & benchmark ===")
    for name, text in _build_tests():
        t0 = time.time()
        encoded, codes = encode_text(text)
        t1 = time.time()

        ok_prefix = _is_prefix_free(codes)
        ok_order = _dict_order_ok(text, codes)
        ok_canon = _canonical_ok(codes) if len(codes) > 1 else True
        enc2, codes2 = encode_text(text)
        ok_det = (encoded == enc2 and codes == codes2)
        ok_len = (len(encoded) == sum(len(codes[ch]) for ch in text))
        try:
            dec = _decode_for_test(encoded, codes)
            ok_round = (dec == text)
        except Exception:
            ok_round = False

        print(f"\n[CASE] {name} | n={len(text)}, s={len(set(text))}, time_ms={round((t1 - t0) * 1000, 3)}")
        print(" prefix_free  :", ok_prefix)
        print(" dict_order   :", ok_order)
        print(" canonical    :", ok_canon)
        print(" deterministic:", ok_det)
        print(" len_match    :", ok_len)
        print(" round_trip   :", ok_round)
        if not all([ok_prefix, ok_order, ok_canon, ok_det, ok_len, ok_round]):
            print(" CODES:", codes)
            print(" ENCODED (first 128b):", encoded[:128], "..." if len(encoded) > 128 else "")

    # --------------------------
    # Q1B: rigorous decoding suite (run once)
    # --------------------------
    print("\n=== Q1B: decode_text — strict validation suite ===")
    results = run_strict_suite()
    _print_table(results)
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    print(f"\nSUMMARY: {passed}/{total} cases passed")
    if passed == total:
        print("All Q1B tests passed")
    else:
        print("Some Q1B tests failed  (see 'exception' column for details)")

    # --------------------------
    # Q2A/Q2B: tree isomorphism tests (as before)
    # --------------------------
    random.seed(42)

    # Random permutation tests
    print("\n" + "=" * 70)
    print("Random permutation tests (Q2):")
    edges1 = [(0,1),(1,2),(1,3),(3,4),(3,5),(2,6)]  # n=7 (a small broom)
    n = 7
    T1_perm_base = mat_from_edges(n, edges1)

    for trial in range(3):
        perm = list(range(n))
        random.shuffle(perm)
        edges2 = permute_edges(n, edges1, perm)
        T2_perm = mat_from_edges(n, edges2)

        print(f"-- Trial {trial+1} perm = {perm}")
        iso = are_isomorphic(T1_perm_base, T2_perm)
        print(f"are_isomorphic (Section 2A): {iso}")
        mapping = mapping_isomorphic(T1_perm_base, T2_perm)
        print("mapping_isomorphic (Section 2B):")
        print_mapping(mapping)

        assert iso, "Expected True for permuted labels."
        assert mapping, "Expected non-empty mapping for permuted labels."
        assert validate_isomorphism(matrix_to_adj(T1_perm_base), matrix_to_adj(T2_perm), mapping), \
               "Permutation mapping failed adjacency validation."

        base_edges = edges_from_matrix(T1_perm_base)
        mapped_edges = edges_apply_mapping(n, base_edges, mapping)
        target_edges = edges_canonical_from_matrix(T2_perm)
        assert mapped_edges == target_edges, "Mapped edges != target edges of T2."

    # Invalid input tests
    print("\n" + "=" * 70)
    print("Invalid input tests (Q2):")
    M_bad_self = [
        [False, True ],
        [ True, True ]  # self-loop at (1,1)
    ]
    try:
        _ = are_isomorphic(M_bad_self, M_bad_self)
        print("ERROR: expected ValueError for self-loop matrix (2A).")
    except ValueError as e:
        print(f"Caught expected error (2A, self-loop): {e}")

    try:
        _ = mapping_isomorphic(M_bad_self, M_bad_self)
        print("ERROR: expected ValueError for self-loop matrix (2B).")
    except ValueError as e:
        print(f"Caught expected error (2B, self-loop): {e}")

    M_bad_asym = [
        [False, True ],
        [False, False]
    ]
    try:
        _ = are_isomorphic(M_bad_asym, M_bad_asym)
        print("ERROR: expected ValueError for non-symmetric matrix (2A).")
    except ValueError as e:
        print(f"Caught expected error (2A, non-symmetric): {e}")

    try:
        _ = mapping_isomorphic(M_bad_asym, M_bad_asym)
        print("ERROR: expected ValueError for non-symmetric matrix (2B).")
    except ValueError as e:
        print(f"Caught expected error (2B, non-symmetric): {e}")

    # Named cases + strict checks
    cases = []

    T1 = mat_from_edges(1, [])
    T2 = mat_from_edges(1, [])
    cases.append(("Single node", T1, T2))

    T1 = mat_from_edges(5, [(0,1),(0,2),(0,3),(0,4)])
    T2 = mat_from_edges(5, [(3,0),(3,1),(3,2),(3,4)])
    cases.append(("Star K1,4 permuted", T1, T2))

    T1 = mat_from_edges(5, [(0,1),(1,2),(2,3),(3,4)])
    T2 = mat_from_edges(5, [(2,1),(1,0),(0,3),(3,4)])
    cases.append(("Path (5 nodes) permuted", T1, T2))

    T1 = mat_from_edges(5, [(0,1),(0,2),(0,3),(0,4)])
    T2 = mat_from_edges(5, [(0,1),(1,2),(2,3),(3,4)])
    cases.append(("Star vs Path (n=5)", T1, T2))

    T1 = mat_from_edges(6, [(0,1),(1,2),(2,3),(3,4),(4,5)])
    T2 = mat_from_edges(6, [(5,4),(4,3),(3,2),(2,1),(1,0)])
    cases.append(("Even path (6) reversed", T1, T2))

    for name, A, B in cases:
        print("\n" + "=" * 70)
        print(f"Case: {name}")

        # --- Q2A: test are_isomorphic ---
        iso = are_isomorphic(A, B)
        print(f"are_isomorphic (Section 2A): {iso}")

        # --- Q2B: test mapping_isomorphic ---
        mapping = mapping_isomorphic(A, B)
        print("mapping_isomorphic (Section 2B):")
        print_mapping(mapping)

        if iso:
            assert mapping, "Expected a non-empty mapping for isomorphic trees."
            adj1 = matrix_to_adj(A)
            adj2 = matrix_to_adj(B)
            assert validate_isomorphism(adj1, adj2, mapping), "Mapping failed validation."
        else:
            assert mapping == [], "Expected empty mapping for non-isomorphic trees."

    print("\n" + "=" * 70)
    print("All strict checks passed across Q1 & Q2. Done.")


if __name__ == "__main__":
    main()