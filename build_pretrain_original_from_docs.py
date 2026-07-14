import argparse
import json
import re
import unicodedata
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import config as cfg
from tqdm import tqdm


SUPPORTED_SUFFIXES = {".txt", ".text", ".docx", ".pdf", ".doc"}

DEFAULT_INPUT_DIR = cfg.DOCS_INPUT_DIR
DEFAULT_OUTPUT_PATH = cfg.DOCS_PRETRAIN_OUTPUT_PATH
DEFAULT_HAPPY_PATH = cfg.PRETRAIN_HAPPY_ORIGIN_DATA
DEFAULT_MERGED_PATH = cfg.PRETRAIN_ORIGIN_DATA

MIN_CHARS = 30
MAX_SEQ = cfg.get_active_config(stage="pretrain", mode="train")["MAX_SEQ_LEN"]
CAPTION_MAX_LEN = MIN_CHARS

NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
W_NS = NS["w"]


def w_tag(name: str) -> str:
    return f"{{{W_NS}}}{name}"


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u3000", " ")
    text = re.sub(r"[\ue000-\uf8ff\ufffd]", "", text)
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([,.;:!?%。，！？；：、）\]\}])", r"\1", text)
    text = re.sub(r"([（\[\{])\s+", r"\1", text)
    return text.strip()


def paragraph_style_id(paragraph) -> str:
    style = paragraph.find("./w:pPr/w:pStyle", NS)
    if style is None:
        return ""
    return style.attrib.get(w_tag("val"), "")


def is_toc_paragraph(paragraph) -> bool:
    style_id = paragraph_style_id(paragraph).lower()
    if style_id.startswith("toc"):
        return True

    instr_text = " ".join(
        node.text or "" for node in paragraph.iter(w_tag("instrText"))
    ).upper()
    return "TOC" in instr_text


def is_caption_text(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return True

    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) > CAPTION_MAX_LEN:
        return False

    chinese_caption = re.match(
        r"^(图|表)\s*[\d一二三四五六七八九十百]+([\s.．:：、\-—]|$)",
        compact,
    )
    english_caption = re.match(
        r"^(fig\.?|figure|table)\s*[0-9ivxlcdmIVXLCDM]+([\s.．:：、\-—]|$)",
        compact,
        flags=re.IGNORECASE,
    )
    return bool(chinese_caption or english_caption)


def is_noise_text(text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return True

    compact = re.sub(r"\s+", "", text)
    if compact in {"目录", "目錄", "目次"}:
        return True
    if is_caption_text(text):
        return True
    if re.fullmatch(r"(第?\d+页?)", compact):
        return True
    if re.fullmatch(r"page\d+", compact, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"[-—–]?\d+[-—–]?", compact):
        return True
    if re.fullmatch(r"[\-—–·•●○▪■。…]+", compact):
        return True
    if re.search(r"\.{3,}\s*\d+$", text):
        return True
    return False


def is_hidden_run(node) -> bool:
    return (
        local_name(node.tag) == "r"
        and node.find("./w:rPr/w:vanish", NS) is not None
    )


def extract_text_from_node(node) -> str:
    parts = []

    for child in list(node):
        tag = local_name(child.tag)
        if tag in {
            "instrText",
            "fldChar",
            "delText",
            "del",
            "moveFrom",
            "tbl",
            "drawing",
            "pict",
            "object",
        }:
            continue
        if is_hidden_run(child):
            continue
        if tag == "t" and child.text:
            parts.append(child.text)
            continue
        if tag == "tab":
            parts.append(" ")
            continue
        if tag in {"br", "cr"}:
            parts.append("\n")
            continue
        parts.append(extract_text_from_node(child))

    return "".join(parts)


def iter_body_paragraphs_without_tables(element):
    for child in list(element):
        tag = local_name(child.tag)
        if tag == "tbl":
            continue
        if tag == "p":
            yield child
            continue
        yield from iter_body_paragraphs_without_tables(child)


def read_docx_paragraphs(path: Path) -> tuple[list[str], int]:
    try:
        with zipfile.ZipFile(path) as docx_zip:
            xml_bytes = docx_zip.read("word/document.xml")
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"not a valid docx file: {path}") from exc
    except KeyError as exc:
        raise RuntimeError(f"missing word/document.xml in docx: {path}") from exc

    root = ElementTree.fromstring(xml_bytes)
    body = root.find("w:body", NS)
    if body is None:
        return [], 0

    paragraphs = []
    filtered_count = 0
    for paragraph in iter_body_paragraphs_without_tables(body):
        raw_text = normalize_text(extract_text_from_node(paragraph))
        if not raw_text:
            continue
        if is_toc_paragraph(paragraph) or is_noise_text(raw_text):
            filtered_count += 1
            continue
        paragraphs.append(raw_text)

    return paragraphs, filtered_count


def read_text_file(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = path.read_text(encoding=encoding)
            return [part for part in re.split(r"\n\s*\n", text) if part.strip()]
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"could not decode text file: {path}")


def read_pdf_text(path: Path) -> list[str]:
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF support requires pypdf or PyPDF2. Install one of them, "
                "or convert the PDF to txt/docx first."
            ) from exc

    reader = PdfReader(str(path))
    paragraphs = []
    for page in reader.pages:
        text = page.extract_text() or ""
        paragraphs.extend(part for part in re.split(r"\n\s*\n", text) if part.strip())
    return paragraphs


def read_doc_text(path: Path) -> list[str]:
    try:
        import win32com.client
    except ImportError as exc:
        raise RuntimeError(
            "legacy .doc support requires Microsoft Word automation via pywin32. "
            "Install pywin32/Microsoft Word, or convert the file to .docx or .txt."
        ) from exc

    word = None
    document = None
    try:
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        document = word.Documents.Open(str(path.resolve()), ReadOnly=True)
        text = document.Content.Text or ""
        return [part for part in re.split(r"\n\s*\n", text) if part.strip()]
    finally:
        if document is not None:
            document.Close(False)
        if word is not None:
            word.Quit()


def read_document_paragraphs(path: Path) -> tuple[list[str], int]:
    suffix = path.suffix.lower()
    if suffix == ".docx":
        return read_docx_paragraphs(path)
    if suffix in {".txt", ".text"}:
        return read_text_file(path), 0
    if suffix == ".pdf":
        return read_pdf_text(path), 0
    if suffix == ".doc":
        return read_doc_text(path), 0
    raise RuntimeError(f"unsupported file type: {suffix}")


def paragraphs_to_records(paragraphs: list[str]) -> tuple[list[str], int]:
    if MAX_SEQ <= 0:
        raise ValueError(f"MAX_SEQ must be positive, got {MAX_SEQ}")

    records = []
    buffer = []
    filtered_count = 0

    def buffer_text() -> str:
        return normalize_text("\n".join(buffer))

    def flush_buffer() -> None:
        nonlocal buffer, filtered_count
        text = buffer_text()
        buffer = []
        if not text:
            return
        if len(text) >= MIN_CHARS:
            records.append(text)
        else:
            filtered_count += 1

    for paragraph in paragraphs:
        paragraph = normalize_text(paragraph)
        if is_noise_text(paragraph):
            filtered_count += 1
            continue
        if len(paragraph) >= MAX_SEQ:
            flush_buffer()
            records.append(paragraph)
            continue
        buffer.append(paragraph)
        if len(buffer_text()) >= MAX_SEQ:
            flush_buffer()

    flush_buffer()
    return records, filtered_count


def iter_input_files(input_path):
    input_path = Path(input_path)
    if not input_path.exists():
        print(f"Warning: docs input path does not exist: {input_path}")
        return

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise RuntimeError(f"unsupported file type: {input_path.suffix}")
        yield input_path
        return

    yield from (
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )


def write_jsonl_record(output_file, text: str) -> bool:
    text = normalize_text(text)
    if not text:
        return False
    output_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    return True


def build_docs_pretrain_jsonl(input_dir=None, output_path=None) -> None:
    input_dir = input_dir or DEFAULT_INPUT_DIR
    output_path = Path(output_path or DEFAULT_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_files = list(iter_input_files(input_dir))
    total_files = 0
    skipped_files = 0
    total_paragraphs = 0
    total_records = 0
    total_filtered = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        progress = tqdm(
            input_files,
            total=len(input_files),
            desc="Processing documents",
            unit="file",
            dynamic_ncols=True,
        )

        for file_path in progress:
            total_files += 1
            progress.set_postfix_str(f"file={file_path.name}")
            try:
                paragraphs, extract_filtered = read_document_paragraphs(file_path)
                records, merge_filtered = paragraphs_to_records(paragraphs)
                kept_records = 0
                for record in records:
                    if write_jsonl_record(output_file, record):
                        kept_records += 1

                total_paragraphs += len(paragraphs)
                total_records += kept_records
                total_filtered += extract_filtered + merge_filtered
                tqdm.write(
                    f"  {file_path.name}: paragraphs={len(paragraphs)} "
                    f"kept={kept_records} "
                    f"filtered={extract_filtered + merge_filtered}"
                )
            except Exception as exc:
                skipped_files += 1
                tqdm.write(f"Warning: skipped {file_path}: {exc}")

    tqdm.write(f"docs_input_dir: {input_dir}")
    tqdm.write(f"docs_output_path: {output_path}")
    tqdm.write(f"MIN_CHARS: {MIN_CHARS}")
    tqdm.write(f"MAX_SEQ: {MAX_SEQ}")
    tqdm.write(f"total_files: {total_files}")
    tqdm.write(f"skipped_files: {skipped_files}")
    tqdm.write(f"total_paragraphs: {total_paragraphs}")
    tqdm.write(f"total_records: {total_records}")
    tqdm.write(f"total_filtered: {total_filtered}")


def iter_text_jsonl_records(path: Path, source_name: str, limit=None):
    if not path.exists():
        raise FileNotFoundError(f"{source_name} data not found: {path}")

    yielded_count = 0
    with path.open("r", encoding="utf-8") as source_file:
        for line_num, line in enumerate(source_file, 1):
            if limit is not None and yielded_count >= limit:
                break

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"Warning: skip invalid JSON in {path} line {line_num}: {exc}"
                )
                continue

            text = normalize_text(str(item.get("text", "")))
            if not text:
                continue
            yielded_count += 1
            yield {"text": text}


def merge_pretrain_sources(happy_path=None, docs_path=None, output_path=None) -> None:
    happy_path = Path(happy_path or DEFAULT_HAPPY_PATH)
    docs_path = Path(docs_path or DEFAULT_OUTPUT_PATH)
    output_path = Path(output_path or DEFAULT_MERGED_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_limits = cfg.get_pretrain_original_source_limits()
    source_specs = [
        ("Happy-LLM original data", happy_path, source_limits["happy"]),
        ("document supplemental data", docs_path, source_limits["docs"]),
    ]
    counts = {}

    mode_label = cfg.CONFIG_MODE
    print(f"CONFIG_MODE: {mode_label}")
    if mode_label == "test":
        print(
            "Test mode pretrain source limits: "
            f"happy={source_limits['happy']}, docs={source_limits['docs']}"
        )
    else:
        print("Train mode pretrain source limits: use all available data.")

    with output_path.open("w", encoding="utf-8") as output_file:
        for source_name, source_path, source_limit in source_specs:
            count = 0
            for record in iter_text_jsonl_records(
                source_path,
                source_name,
                limit=source_limit,
            ):
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
            counts[source_name] = count
            print(
                f"Merged {count} records from {source_name}: {source_path}; "
                f"limit={source_limit if source_limit is not None else 'all'}"
            )

    print(f"Regenerated merged pretrain source: {output_path}")
    print(
        "pretrain_original.jsonl = "
        "pretrain_original_happy.jsonl + pretrain_original_from_docs.jsonl"
    )
    print(f"merge_counts: {counts}")


def build_pretrain_original_from_docs(
    input_dir=None,
    docs_output_path=None,
    happy_path=None,
    merged_output_path=None,
) -> None:
    print("Pretrain source: Happy-LLM original data plus document data.")
    print(f"Happy-LLM original data: {happy_path or DEFAULT_HAPPY_PATH}")
    print(f"Document supplemental data input: {input_dir or DEFAULT_INPUT_DIR}")
    print(f"Document supplemental JSONL: {docs_output_path or DEFAULT_OUTPUT_PATH}")
    print(f"Merged pretrain source: {merged_output_path or DEFAULT_MERGED_PATH}")

    build_docs_pretrain_jsonl(
        input_dir=input_dir or DEFAULT_INPUT_DIR,
        output_path=docs_output_path or DEFAULT_OUTPUT_PATH,
    )
    merge_pretrain_sources(
        happy_path=happy_path or DEFAULT_HAPPY_PATH,
        docs_path=docs_output_path or DEFAULT_OUTPUT_PATH,
        output_path=merged_output_path or DEFAULT_MERGED_PATH,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build and merge pretrain JSONL data from project documents."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory or supported document file. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--docs-output-path",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Document JSONL output path. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--happy-path",
        default=DEFAULT_HAPPY_PATH,
        help=f"Happy-LLM JSONL input path. Default: {DEFAULT_HAPPY_PATH}",
    )
    parser.add_argument(
        "--merged-output-path",
        default=DEFAULT_MERGED_PATH,
        help=f"Merged JSONL output path. Default: {DEFAULT_MERGED_PATH}",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    build_pretrain_original_from_docs(
        input_dir=args.input_dir,
        docs_output_path=args.docs_output_path,
        happy_path=args.happy_path,
        merged_output_path=args.merged_output_path,
    )


if __name__ == "__main__":
    main()
