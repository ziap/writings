#/bin/sh -e

for file in *.typ; do
  base=$(basename -s .typ $file)
  out_pdf="out/$base.pdf"

  echo "Compiling $file -> $out_pdf"
  typst compile --font-path template/fonts/ "$file" "$out_pdf"
done
