# 🚗 Vehicle Specification Extractor (RAG)

<p align="center">
  <strong>Extract vehicle specifications from large service manuals using Retrieval-Augmented Generation (RAG) and rule-based extraction.</strong>
</p>

<p align="center">
  <a href="https://vehicle-specification-extractor-uqnpzdnlmhgpigkqfc59mf.streamlit.app/"><strong>🚀 Live Demo</strong></a>
  &nbsp; • &nbsp;
  <a href="#-installation"><strong>⚙️ Installation</strong></a>
  &nbsp; • &nbsp;
  <a href="#-how-it-works"><strong>🧠 How It Works</strong></a>
</p>

---

## 👨‍💻 Project Information

| Detail | Information |
|---|---|
| **Project Title** | Vehicle Specification Extractor |
| **Developer** | Souvik Das |
| **Institution** | National Institute of Technology Delhi (NIT Delhi) |
| **Department** | Computer Science and Engineering |
| **Technology** | Python, RAG, SBERT, BM25, Streamlit |
| **Extraction Method** | Offline rule-based extraction |
| **Deployment** | Streamlit Community Cloud |

## 🌐 Live Demo

**Try the application here:**

👉 https://vehicle-specification-extractor-uqnpzdnlmhgpigkqfc59mf.streamlit.app/

Ask a question such as:

`Torque for brake caliper bolts`

The application retrieves relevant passages from the vehicle service manual and displays structured specifications with their source sections.

> Note: The live demo depends on the deployed application's availability and the sample manual being accessible.

## 📌 Overview

Vehicle service manuals contain hundreds of pages of technical information, including torque specifications, component measurements, fluid capacities, and repair procedures.

Finding a particular specification manually can be time-consuming.

The **Vehicle Specification Extractor** simplifies this process by allowing users to ask natural-language questions and retrieve relevant technical information from a service manual.

The project uses a hybrid retrieval pipeline combining semantic search and keyword-based ranking, followed by offline rule-based extraction.

For the supplied sample manual, the application reports **852 pages and 898 chunks**.

## ✨ Key Features

- 🔍 **Natural-language queries:** Ask questions in simple English.
- 🧠 **Hybrid retrieval:** Combines dense semantic retrieval with BM25 keyword search and Reciprocal Rank Fusion (RRF).
- 📄 **Section-aware chunking:** Preserves relevant manual sections and specification tables.
- 🛠️ **Rule-based extraction:** Extracts component names, specification types, values, and units.
- 📊 **Structured output:** Displays specifications in a readable table and supports JSON/CSV output.
- 📚 **Source-aware results:** Associates specifications with their manual sections.
- 🔒 **No paid API required:** The default extraction pipeline works offline without an OpenAI API key.
- 🖥️ **Interactive interface:** Built with Streamlit.

## 🧪 Example

**Question:**

`Torque for brake caliper bolts`

**Example output:**

| Component | Specification | Value | Unit | Section |
|---|---|---:|---|---|
| Brake caliper guide pin bolts | Torque | 37 | Nm | 206-03: Front Disc Brake |
| Brake caliper guide pin bolts | Torque | 33 | Nm | 206-04: Rear Disc Brake |
| Brake caliper flow bolt | Torque | 35 | Nm | 206-03: Front Disc Brake |
| Brake caliper flow bolt | Torque | 35 | Nm | 206-04: Rear Disc Brake |

These example values are taken from the supplied manual's extracted results. The front and rear specifications are kept separate because similar components can have different values.

The complete output can also contain other relevant fasteners, such as anchor plate and support bracket bolts.

**Example JSON:**

```json
[
  {
    "component": "Brake caliper guide pin bolts",
    "spec_type": "Torque",
    "value": "37",
    "unit": "Nm",
    "section": "206-03: Front Disc Brake"
  }
]
```

The `section` field provides useful source context. Use the `--minimal` option to omit it when only the four requested fields are needed.

## 🧠 How It Works

```text
Vehicle Service Manual
          |
          v
  Text Extraction & Cleaning
          |
          v
    Section-Aware Chunking
          |
          v
  Embedding & Search Index
          |
          v
 Hybrid Retrieval (SBERT + BM25)
          |
          v
  Relevant Evidence Selection
          |
          v
 Offline Rule-Based Extraction
          |
          v
 Structured Specifications
      (JSON / CSV / UI)
```

### Pipeline Components

| Step | Module | Responsibility |
|---|---|---|
| 1. Text extraction | `src/pdf_extract.py` | Extracts and cleans text from supported document formats. |
| 2. Chunking | `src/chunking.py` | Creates section-aware chunks and preserves specification tables. |
| 3. Embeddings | `src/embeddings.py` | Supports semantic representations and a TF-IDF/LSA fallback. |
| 4. Retrieval | `src/retriever.py` | Combines dense retrieval, BM25, and RRF ranking. |
| 5. Extraction | `src/extractor.py` | Identifies supported specifications from retrieved evidence. |
| 6. Output | `src/main.py` | Produces structured results in JSON or CSV format. |
| 7. Interface | `app.py` | Provides the Streamlit application. |

## 🛠️ Technology Stack

- **Language:** Python
- **Semantic retrieval:** Sentence-BERT (SBERT)
- **Keyword retrieval:** BM25
- **Ranking:** Reciprocal Rank Fusion (RRF)
- **Numerical processing:** NumPy, scikit-learn
- **Document processing:** PyMuPDF, with supported fallback parsers
- **Vector search:** FAISS when available, otherwise NumPy cosine similarity
- **Frontend:** Streamlit
- **Output formats:** JSON and CSV
- **Testing:** Python tests and pytest

## ⚙️ Installation

### Prerequisites

- Python 3.10 or a compatible version supported by the dependencies
- pip
- A vehicle service manual in a supported format

### 1. Clone the repository

```bash
git clone https://github.com/souvik001122/vehicle-specification-extractor.git
cd vehicle-specification-extractor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run a query

```bash
python -m src.main --query "Torque for brake caliper bolts"
```

### 4. Process multiple questions

```bash
python -m src.main --queries-file queries.txt --out outputs/results.csv
```

To generate JSON output:

```bash
python -m src.main --queries-file queries.txt --out outputs/results.json
```

### 5. Launch the Streamlit interface

```bash
streamlit run app.py
```

Open the local URL printed in the terminal to use the application.

### 6. Run tests

```bash
python tests/test_pipeline.py
```

Alternatively, if pytest is installed:

```bash
pytest -q
```

## 📁 Project Structure

```text
vehicle-specification-extractor/
├── app.py
├── config.py
├── queries.txt
├── requirements.txt
├── README.md
├── data/
│   └── sample-service-manual.txt
├── outputs/
│   ├── results.json
│   ├── results.csv
│   └── example_minimal.json
├── src/
│   ├── pdf_extract.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── extractor.py
│   └── main.py
└── tests/
    └── test_pipeline.py
```

*The exact files may vary slightly with the repository version.*

## 🎯 Design Decisions

### Section-aware chunking

Vehicle manuals often repeat component names across different sections. Preserving section metadata helps distinguish front brake specifications from rear brake specifications.

### Hybrid retrieval

Dense embeddings help identify semantically related passages, while BM25 helps match exact component names, units, and technical terms. RRF combines their rankings.

### Evidence-grounded extraction

The rule-based extractor extracts values from retrieved manual content rather than generating specifications from outside knowledge.

### Structured output

The application returns component names, specification types, values, units, and source sections so results are easier to inspect and reuse.

## ⚠️ Limitations and Future Improvements

- Extraction accuracy depends on the quality and structure of the source manual.
- Complex staged torque procedures and angle specifications may require additional parsing rules.
- Some fluid-capacity tables may produce extra or incomplete rows.
- Retrieval can return less relevant passages for broad or ambiguous queries.
- Vehicle configuration differences, such as engine or drivetrain variants, are not necessarily resolved automatically.
- A labelled evaluation dataset is needed to measure retrieval precision, recall, and extraction accuracy systematically.

Potential improvements include cross-encoder reranking, more robust table parsing, vehicle-variant filtering, page-level citations, and expanded evaluation tests.

## 🔐 Privacy and Safety

- The default pipeline does not require a paid API or API key.
- Do not commit passwords, credentials, or private configuration files.
- Publish only service manuals you have permission to redistribute.
- Treat extracted technical specifications as information to verify against the applicable vehicle and official service manual before performing repairs.

## 👨‍🎓 About the Developer

**Souvik Das**  
B.Tech, Computer Science and Engineering  
**National Institute of Technology Delhi (NIT Delhi)**

This project demonstrates the application of information retrieval, natural-language processing, document processing, and structured information extraction to a practical automotive use case.

---

## ⭐ Acknowledgements

Developed as an academic project exploring retrieval-based information extraction from technical documentation.

If you find this project useful, consider starring the repository on GitHub.
