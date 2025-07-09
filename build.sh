#/bin/sh -e

unset SOURCE_DATE_EPOCH

for file in src/*.typ; do
  base=$(basename -s .typ $file)
  out_pdf="out/$base.pdf"

  echo "Compiling $file -> $out_pdf"
  typst compile --font-path src/template/fonts/ "$file" "$out_pdf"
done
