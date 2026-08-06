# Kaggle solution post

`kaggle_solution_post.md` is the drop-in-ready Kaggle Discussions article for
Jiwei Liu's contribution to Team Kaggle Agent's ROGII solution.

The article follows the publishing contract introduced by
[`NVIDIA/nvidia-kaggle#16`](https://github.com/NVIDIA/nvidia-kaggle/pull/16):

- every image reference is an absolute public raw-content URL;
- every referenced visual is checked into `writeup/plots/`;
- the URL path preserves the repository-relative plot path; and
- the final Markdown uses only Kaggle-compatible tables, fenced blocks, and
  image syntax.

The image base URL is:

```text
https://raw.githubusercontent.com/daxiongshu/rogii/main/writeup
```

Because the repository remains private during the competition, GitHub returns
404 for these URLs today. Once the repository is public and this commit is on
`main`, run the PR #16 exporter/verifier against the article before pasting it
into Kaggle:

```bash
python kaggle_markdown_export.py \
  writeup/kaggle_solution_post.md \
  --base-url https://raw.githubusercontent.com/daxiongshu/rogii/main/writeup \
  --output /tmp/kaggle_solution_post.verified.md \
  --verify
```

The exporter should report zero rewritten image references because this file
is already in publication form; `--verify` performs the required network and
image-content checks.
