import json

from tqdm import tqdm


def split_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


input_file = "data/mobvoi_seq_monkey_general_open_corpus.jsonl"
output_file = "data/seq_monkey_datawhale.jsonl"

bad_lines = 0
written_chunks = 0
total_lines = count_lines(input_file)

with open(output_file, "w", encoding="utf-8") as pretrain:
    with open(input_file, "r", encoding="utf-8") as f:
        progress = tqdm(
            f,
            total=total_lines,
            desc="Processing",
            unit="line",
            dynamic_ncols=True,
        )

        for line_num, line in enumerate(progress, start=1):
            try:
                sample = json.loads(line)
                text = sample["text"]
            except json.JSONDecodeError as e:
                bad_lines += 1
                progress.set_postfix(bad=bad_lines, chunks=written_chunks)
                tqdm.write(f"skip bad json: line={line_num}, error={e}")
                continue
            except KeyError:
                bad_lines += 1
                progress.set_postfix(bad=bad_lines, chunks=written_chunks)
                tqdm.write(f"skip missing text field: line={line_num}")
                continue

            for chunk in split_text(text, chunk_size=512):
                chunk = chunk.strip()
                if chunk:
                    pretrain.write(
                        json.dumps({"text": chunk}, ensure_ascii=False) + "\n"
                    )
                    written_chunks += 1

            if line_num % 1000 == 0:
                progress.set_postfix(bad=bad_lines, chunks=written_chunks)

print(f"done. bad_lines={bad_lines}, written_chunks={written_chunks}")
