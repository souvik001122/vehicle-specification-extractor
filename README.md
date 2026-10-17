# 🚗 Vehicle Specification Extractor (RAG)

<p align="center">
  <strong>Extract structured vehicle specifications from service manuals using hybrid retrieval and rule-based information extraction.</strong>
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
| **Language** | Python |
| **Retrieval** | Semantic Search, BM25, Reciprocal Rank Fusion (RRF) |
| **Extraction** | Offline Rule-Based Extraction |
| **Interface** | Streamlit |
| **Deployment** | Streamlit Community Cloud |

## 🌐 Live Demo

Try the application:

**[Open Vehicle Specification Extractor](https://vehicle-specification-extractor-uqnpzdnlmhgpigkqfc59mf.streamlit.app/)**

Example query:

`Torque for brake caliper bolts`

The application retrieves relevant passages from a vehicle service manual and extracts structured specifications with component names, values, units, and source sections.

The default extraction pipeline works offline and does not require an OpenAI API key.

## 📌 Overview

Vehicle workshop manuals contain hundreds of pages of technical information, including torque specifications, component measurements, fluid capacities, and repair procedures.

Finding a specific value manually can be time-consuming, especially when similar components appear in different sections of a manual.

The **Vehicle Specification Extractor** addresses this problem through a Retrieval-Augmented Generation (RAG)-style pipeline that combines hybrid document retrieval with rule-based extraction.

The system retrieves relevant evidence and extracts specifications from the manual rather than relying on unsupported generated values.

The included sample data is a text extraction of a 2014 Ford F-150 Workshop Manual.

## ✨ Key Features

- 🔍 **Natural-language queries:** Ask questions in simple English.
- 🧠 **Hybrid retrieval:** Combines dense semantic retrieval with BM25 keyword ranking and Reciprocal Rank Fusion.
- 📄 **Section-aware chunking:** Preserves manual section metadata and specification tables.
- 🔧 **Torque-table extraction:** Extracts torque values and units from supported table layouts.
- 🛠️ **Procedure extraction:** Recognizes supported torque instructions, measurements, and fluid capacities.
- 📑 **PDF and text support:** Processes supported PDF and TXT manuals.
- 🔄 **Improved PDF table handling:** Handles certain torque tables where component names, numeric values, and dash placeholders are split across separate text lines.
- 📊 **Structured output:** Produces readable results and supports JSON and CSV output.
- 📚 **Source-aware results:** Associates extracted specifications with their source sections.
- 🔒 **No paid API required by default:** The standard rule-based extraction pipeline runs without an API key.
- 🖥️ **Interactive interface:** Provides a Streamlit interface for uploading manuals and querying specifications.
- 🧪 **Regression testing:** Includes tests for front/rear brake-caliper torque extraction and flattened PDF-style torque tables.

## 🧪 Example Query and Output

**Query:**

`Torque for brake caliper bolts`

**Example results from the supplied manual:**

| Component | Specification | Value | Unit | Section |
|---|---|---:|---|---|
| Brake caliper guide pin bolts | Torque | 37 | Nm | 206-03: Front Disc Brake |
| Brake caliper guide pin bolts | Torque | 33 | Nm | 206-04: Rear Disc Brake |
| Brake caliper flow bolt | Torque | 35 | Nm | 206-03: Front Disc Brake |
| Brake caliper flow bolt | Torque | 35 | Nm | 206-04: Rear Disc Brake |
| Brake caliper anchor plate bolts | Torque | 250 | Nm | 206-03: Front Disc Brake |
| Brake caliper support bracket bolts | Torque | 150 | Nm | 206-04: Rear Disc Brake |

These values illustrate why source-section information matters: similar components can have different specifications depending on their location.

The exact results returned by the application depend on the manual, its formatting, retrieval quality, and the query.

### Example JSON

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

The default output includes the `section` field to provide source context. The `--minimal` option can be used to return only the four core fields: `component`, `spec_type`, `value`, and `unit`.

## 🧠 How It Works

```text
Vehicle Service Manual (PDF / TXT)
                 |
                 v
       Text Extraction & Cleaning
                 |
                 v
          Table Detection
                 |
                 v
         Section-Aware Chunking
                 |
                 v
       Embedding & Search Index
                 |
                 v
       Hybrid Retrieval (Dense
             + BM25)
                 |
                 v
       Reciprocal Rank Fusion
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
| 1. Document processing | `src/pdf_extract.py` | Extracts and cleans text from supported document formats. |
| 2. Table parsing | `src/tables.py` | Detects and parses supported torque, specification, material, and parts tables. |
| 3. Chunking | `src/chunking.py` | Creates section-aware chunks and preserves table structure where supported. |
| 4. Embeddings | `src/embeddings.py` | Supports semantic embeddings and a TF-IDF/LSA fallback. |
| 5. Retrieval | `src/retriever.py` | Combines dense retrieval, BM25, and Reciprocal Rank Fusion. |
| 6. Extraction | `src/extractor.py` | Extracts supported specifications from retrieved evidence using rule-based parsing. |
| 7. Output | `src/main.py` | Produces structured results and supports JSON/CSV output. |
| 8. Interface | `app.py` | Provides the Streamlit application and uploaded-manual workflow. |

## 🛠️ Technology Stack

- **Programming language:** Python
- **Semantic retrieval:** Sentence Transformers / SBERT when available
- **Keyword retrieval:** BM25
- **Ranking:** Reciprocal Rank Fusion (RRF)
- **Fallback representations:** TF-IDF and Latent Semantic Analysis (LSA)
- **Numerical processing:** NumPy and scikit-learn
- **Document processing:** PyMuPDF with supported fallback parsers
- **Vector search:** FAISS when available, otherwise NumPy-based cosine similarity
- **Frontend:** Streamlit
- **Output formats:** JSON and CSV
- **Testing:** Python unittest-compatible tests and pytest

## ⚙️ Installation

### Prerequisites

- Python 3.10 or a compatible version supported by the installed dependencies
- pip
- A service manual in a supported format

### 1. Clone the repository

```bash
git clone https://github.com/souvik001122/vehicle-specification-extractor.git
cd vehicle-specification-extractor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run a single query

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

Open the local URL displayed in the terminal.

### 6. Run the test suite

```bash
python tests/test_pipeline.py
```

Alternatively, if pytest is installed:

```bash
pytest -q
```

The included regression tests cover supported extraction behavior, including a flattened PDF-style torque table represented as a text chunk.

## 📁 Project Structure

```text
vehicle-specification-extractor/
├── app.py
├── config.py
├── queries.txt
├── requirements.txt
├── README.md
│
├── data/
│   └── sample-service-manual.txt
│
├── outputs/
│   ├── results.json
│   ├── results.csv
│   └── example_minimal.json
│
├── src/
│   ├── __init__.py
│   ├── pdf_extract.py
│   ├── tables.py
│   ├── textutil.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── extractor.py
│   └── main.py
│
└── tests/
    └── test_pipeline.py
```

The structure above reflects the updated project ZIP. Runtime cache files may be created when the application processes manuals.

## 🎯 Design Decisions

### 1. Section-aware chunking

Vehicle manuals often repeat component names in different sections. Keeping section metadata helps distinguish front and rear brake specifications and improves source traceability.

### 2. Hybrid retrieval

Dense semantic search helps identify passages with related meanings, while BM25 helps match exact technical terms, component names, and units. Reciprocal Rank Fusion combines the rankings.

### 3. Evidence-grounded extraction

The default rule-based extractor parses supported values from retrieved manual content. This reduces reliance on generated answers, although retrieval and parsing errors can still occur.

### 4. Improved torque-table parsing

PDF text extraction can separate component names, numeric cells, and placeholder values across multiple lines. The updated parser handles certain split-line torque-table layouts and can detect torque tables embedded in ordinary text chunks.

### 5. Source-aware specifications

The section field helps users verify extracted values against the relevant part of the service manual.

### 6. Content-aware upload caching

Uploaded manuals are cached using a content-derived identifier so that uploading a different file with the same filename does not automatically reuse the previous file's cached pipeline.

## 🧪 Testing and Validation

The updated project includes **20 automated tests** covering:

- Query parsing and keyword matching
- Text cleaning and section-aware chunking
- Torque extraction for front and rear brake components
- Wheel-nut torque and other supported specifications
- Fluid-capacity and measurement extraction
- JSON and CSV output
- Handling of a flattened PDF-style torque table
- Fallback behavior for unsupported or irrelevant queries

The regression test checks the table-parsing issue where component names and numeric values are represented on separate lines.

Passing these tests does not guarantee perfect extraction from every PDF. Results should be validated against the specific manual and vehicle configuration being used.

## ⚠️ Limitations and Future Improvements

- Extraction quality depends on the source document, PDF text layout, and retrieval quality.
- Complex or scanned tables may require OCR or more layout-aware table extraction.
- Staged torque procedures, such as a torque value followed by an additional rotation angle, may not be fully represented in the current output schema.
- Fluid-capacity tables with wrapped cells may produce extra or incomplete rows.
- Broad or ambiguous queries can retrieve less relevant evidence.
- Vehicle-specific variations, including engine, drivetrain, and trim differences, are not necessarily resolved automatically.
- A labelled evaluation dataset is needed to measure retrieval precision, recall, and extraction accuracy systematically.

Potential future improvements include:

- Layout-aware PDF table extraction
- Improved OCR and scanned-table processing
- Cross-encoder reranking
- Query expansion for component synonyms
- Vehicle-variant filtering
- Page-level citations and evidence highlighting
- Larger evaluation datasets with precision, recall, and F1 metrics

## 🔐 Privacy and Safety

- The default rule-based pipeline does not require a paid API or API key.
- Do not commit credentials, passwords, or private configuration files.
- Publish only service manuals that you have permission to redistribute.
- Verify extracted specifications against the applicable official service manual and vehicle configuration before carrying out repairs.

## 👨‍🎓 About the Developer

**Souvik Das**  
B.Tech, Computer Science and Engineering  
**National Institute of Technology Delhi (NIT Delhi)**

This project demonstrates the application of information retrieval, document processing, natural-language processing, and structured information extraction to a practical automotive use case.

---

## ⭐ Acknowledgements

Developed as an academic project exploring retrieval-based information extraction from technical documentation.

If you find this project useful, consider starring the repository on GitHub.
