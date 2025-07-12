#!/usr/bin/env nu

# Start a writing session of a file
def "main watch" [
  filename: path, # The file to edit
] {
  let base = $filename | path parse
  let out_pdf = $"out/($base.stem).pdf"

  # Generate a typ/pdf file if not already exists
  touch $filename
  typst compile --font-path src/template/fonts/ $filename $out_pdf

  let pid = job spawn { start $out_pdf }
  do --ignore-errors {
    typst watch --font-path src/template/fonts/ $filename $out_pdf
  }

  job kill $pid
}

# Build all typst files in 'src/' to their corresponding pdf files in 'out/'
def main [] {
  # Remove this to avoid typst from setting the current date to 1980/1/1
  if "SOURCE_DATE_EPOCH" in $env {
    hide-env SOURCE_DATE_EPOCH
  }

  ls src/*.typ | each { |file|
    let base = $file.name | path parse
    let out_pdf = $"out/($base.stem).pdf"

    print $"Compiling ($file.name) -> ($out_pdf)"
    typst compile --font-path src/template/fonts/ $file.name $out_pdf
  }

  null
}
