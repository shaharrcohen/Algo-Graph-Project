# 🧮 Graph Algorithms Project — Huffman Coding & Tree Isomorphism

## 📖 Overview
This project presents two classical algorithmic implementations in graph theory and data compression:

1. **Huffman Coding (encode/decode)** — constructs canonical prefix-free Huffman codes, performs encoding and decoding validation, and ensures deterministic ordering of symbols.  
2. **Tree Isomorphism** — efficiently determines whether two trees are isomorphic and constructs a node mapping when an isomorphism exists.

The project highlights **algorithmic correctness**, **complexity proofs**, and **reproducible testing**, showcasing both theoretical understanding and practical implementation skills.

---

## 🧩 Files
| File | Description |
|------|--------------|
| `algo_graph_project.py` | Main implementation containing both Huffman and Tree Isomorphism algorithms. |
| `README.md` | Full project documentation (this file). |
| `.gitignore` | Ignore unnecessary cache, virtual environment, and IDE files. |
| `LICENSE` | MIT License granting open-source usage rights. |

---

## 🧠 Key Features 🧩

| Feature | Description |
|----------|-------------|
| 🟩 **Canonical Huffman Coding** | Builds unique, prefix-free codes ensuring deterministic decoding. |
| ⚙️ **Tree Isomorphism Detection** | Determines structural equality of trees in linear time. |
| 🧪 **Self-Testing Module** | Includes built-in test suites to verify correctness automatically. |
| 🔁 **Deterministic Behavior** | All random processes use seeded generators for reproducibility. |
| 💡 **Readable Structure** | Fully documented, modular, and adheres to clean Python conventions. |
| 📊 **Complexity Analysis** | Each algorithm includes formal proofs and empirical validation. |

---

## 🧮 Complexity Analysis 📈

| Algorithm | Description | Time Complexity | Space Complexity |
|------------|--------------|----------------|------------------|
| **Huffman Encoding** | Builds optimal prefix-free binary tree | **O(n log n)** | **O(n)** |
| **Huffman Decoding** | Reconstructs and decodes encoded strings | **O(n)** | **O(n)** |
| **Tree Isomorphism** | Compares two rooted trees for structural equality | **O(n)** | **O(n)** |

---

## 💻 Requirements 💽

| Requirement | Details |
|--------------|----------|
| 🐍 **Python** | Version 3.10 or higher |
| 📦 **Libraries** | None required — standard library only |
| 🎲 **Random Seeds** | To ensure identical results, `random.seed(0)` is defined inside the code |

---

## 🧾 Example Use Case 📘

| Scenario | Explanation |
|-----------|--------------|
| ✉️ **Encoding messages** | Using Huffman compression for optimal storage. |
| 🌳 **Validating structure** | Checking structural similarity between hierarchies (syntax trees, XML nodes). |
| 🧠 **Algorithmic design patterns** | Demonstrating clean algorithmic architecture for academic and educational use. |

---

## 👩‍💻 Author 🧑‍💻

**Shahar Cohen**  
Mathematics & Computer Science Student — Ariel University  
*Available upon request* ✉️

---

## 🪪 License 📜
Released under the **MIT License** — you are free to use, modify, and distribute with attribution.

---

## 🧭 How to Run
```bash
python src/algo_graph_project.py
