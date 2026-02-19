import camelot
import fitz  # PyMuPDF

file_path = (
    r"C:\projects\Citi Bank rag_reddis\Data\citibank-creditcard-conditions_compress.pdf"
)

# ---- Extract TEXT using PyMuPDF ----
doc = fitz.open(file_path)
for i, page in enumerate(doc):
    text = page.get_text()
    print(f"\n--- Page {i + 1} TEXT ---")
    print(text)

# ---- Extract TABLES using camelot ----
tables = camelot.read_pdf(file_path, pages="all")
print(f"\nTotal tables found: {tables.n}")
for i, table in enumerate(tables):
    df = table.df
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df = df.replace("\n", " ", regex=True)
    print(f"\n--- Table {i + 1} ---")
    print(df)
