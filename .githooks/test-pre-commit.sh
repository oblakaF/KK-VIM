#!/bin/sh
# Self-test for .githooks/pre-commit.
#
# Runs in a temporary repository and never touches the project repository.
# Usage (Git for Windows: Git Bash, or `sh` from PowerShell; Linux/macOS: sh):
#     sh .githooks/test-pre-commit.sh
# Exit code 0 means every case behaved as expected.

hooks_dir=$(cd "$(dirname "$0")" && pwd)
tmp=$(mktemp -d 2>/dev/null || mktemp -d -t kkvim-hook-test)
trap 'rm -rf "$tmp"' EXIT
cd "$tmp" || exit 1

git init -q . || exit 1
git config user.name "hook-test"
git config user.email "hook-test@example.invalid"
git config core.autocrlf false
git config core.hooksPath "$hooks_dir"

failed=0
# expect <accept|reject> <description> <path>...
expect() {
    want=$1
    desc=$2
    shift 2
    git add -f -- "$@" >/dev/null 2>&1
    if git commit -q -m "$desc" >/dev/null 2>&1; then
        got=accept
    else
        got=reject
        git rm -q --cached -- "$@" >/dev/null 2>&1
    fi
    if [ "$got" = "$want" ]; then
        echo "PASS  $want  $desc"
    else
        echo "FAIL  expected $want, got $got: $desc"
        failed=1
    fi
}

mkdir -p docs references/primary scratch sub/scratch
printf '# note\n' > docs/note.md
printf '\211PNG\r\n\032\n' > docs/figure.png
printf '%%PDF-1.4\n' > references/primary/paper.pdf
printf '%%PDF-1.4\n' > UPPER.PDF
printf '%%PDF-1.4\n' > disguised.txt
printf 'AT&TFORM' > book.djvu
printf '\320\317\021\340\241\261\032\341' > chapter.doc
printf 'PK\003\004' > report.docx
printf 'PK\003\004' > bundle.zip
printf 'x' > notes.7z
printf 'page scan' > scratch/page.png
printf 'tmp' > sub/scratch/tmp.txt
printf '# other\n' > docs/other.md

expect accept "plain Markdown document"                 docs/note.md
expect accept "project PNG image outside scratch/"      docs/figure.png
expect reject "force-added PDF in references/"          references/primary/paper.pdf
expect reject "PDF with upper-case extension"           UPPER.PDF
expect reject "PDF content under a .txt name"           disguised.txt
expect reject "DjVu file"                               book.djvu
expect reject "DOC file"                                chapter.doc
expect reject "DOCX file"                               report.docx
expect reject "ZIP archive"                             bundle.zip
expect reject "7z archive"                              notes.7z
expect reject "file in scratch/"                        scratch/page.png
expect reject "file in nested scratch/"                 sub/scratch/tmp.txt
expect reject "allowed file together with a PDF"        docs/other.md UPPER.PDF

if [ "$failed" -eq 0 ]; then
    echo "pre-commit self-test: ALL PASSED"
else
    echo "pre-commit self-test: FAILED"
fi
exit "$failed"
